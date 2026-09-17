"""PyTorch adapter with explicit dtype/device conversion and NumPy-style calls.

Narrow migration reference: ef971a3 torch_backend.py's dtype/device and RNG
mapping. No BackendConfig, distributed, execution-IR or resident dependencies.
"""
import os
import numpy as np

from renormalizer.backend.abstract import AbstractBackend


class TorchBackend(AbstractBackend):
    name = 'torch'
    opt_einsum_name = 'torch'

    def __init__(self, device='cpu'):
        try:
            import torch
        except (ImportError, OSError) as error:
            raise ImportError('PyTorch is not installed or cannot be loaded') from error
        self._torch = torch
        if device != 'cpu' and not (isinstance(device,str) and device.startswith('cuda:') and device[5:].isdigit()):
            raise ValueError(f'unsupported Torch device {device}')
        self._device = torch.device(device)
        if self._device.type == 'cuda' and (not torch.cuda.is_available() or self._device.index >= torch.cuda.device_count()):
            raise ValueError(f'Torch device {device} unavailable')
        self.ndarray = torch.Tensor
        self.device_array_types = (torch.Tensor,)
        self.array_types = (np.ndarray, torch.Tensor)
        self._dtype_map = {np.dtype(name): getattr(torch,name) for name in (
            'bool','int8','uint8','int16','int32','int64','float16','float32','float64','complex64','complex128')}
        self._reverse_dtypes = {value:key for key,value in self._dtype_map.items()}
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(2019)
        self.random = _TorchRandom(self)
        self.linalg = _TorchLinalg(self)
        self.array_namespace = self
        self.memory_errors = (MemoryError, torch.OutOfMemoryError)
        super().__init__()
        if os.environ.get('RENO_FP32') is not None:
            self.use_32bits()

    def current_device(self):
        return str(self._device)

    def owns(self,x):
        return isinstance(x,self._torch.Tensor) and x.device == self._device

    def is_array(self,x):
        return isinstance(x,self.array_types)

    def dtype_of(self,x):
        try:
            return self._reverse_dtypes[x.dtype]
        except KeyError as error:
            raise TypeError(f'unsupported Torch dtype {x.dtype}') from error

    def _dtype(self,dtype):
        if isinstance(dtype,self._torch.dtype):
            return dtype
        return self._dtype_map[np.dtype(dtype)]

    def array(self,data,dtype=None,*,copy=True):
        if copy is not None and type(copy) is not bool:
            raise TypeError('copy must be None, True, or False')
        if dtype is None:
            dtype = self.dtype_of(data) if isinstance(data,self._torch.Tensor) else np.asarray(data).dtype
        dtype = self._dtype(dtype)
        if copy is False:
            if isinstance(data,self._torch.Tensor):
                if not self.owns(data) or data.dtype != dtype:
                    raise ValueError('copy=False cannot transfer or change dtype')
                return data
            if (not isinstance(data,np.ndarray) or self._device.type != 'cpu'
                    or self._dtype(data.dtype) != dtype
                    or any(stride < 0 for stride in data.strides)
                    or not data.flags.writeable):
                raise ValueError('copy=False requires representable shared CPU storage')
        if isinstance(data,np.ndarray) and (any(stride<0 for stride in data.strides) or not data.flags.writeable):
            data=np.array(data,copy=True,order='C')
        result=self._torch.as_tensor(data,dtype=dtype,device=self._device)
        return result.clone() if copy is True else result

    def asarray(self,data,dtype=None):
        return self.array(data,dtype=dtype,copy=None)

    def from_numpy(self,x,*,copy=None):
        return self.array(x,dtype=x.dtype,copy=copy)

    def to_numpy(self,x,*,copy=None):
        if copy is not None and type(copy) is not bool:
            raise TypeError('copy must be None, True, or False')
        if copy is False and (x.device.type!='cpu' or x.is_conj() or x.is_neg()):
            raise ValueError('copy=False cannot transfer or materialize a tensor')
        result=x.detach().resolve_conj().resolve_neg().cpu().numpy()
        return result.copy() if copy is True else result

    def numpy(self,x):
        if x is None:
            return None
        if isinstance(x,np.ndarray):
            return x
        return self.to_numpy(x)

    def strict_call(self,name,*args,**kwargs):
        t=self._torch
        if name=='astype':
            return args[0].to(dtype=self._dtype(args[1]),copy=kwargs.get('copy',True))
        if name in ('zeros','ones','empty','full','arange'):
            if kwargs.get('dtype') is not None:
                kwargs['dtype']=self._dtype(kwargs['dtype'])
            elif name != 'arange':
                kwargs['dtype']=self._dtype(self.real_dtype)
            return getattr(t,name)(*args,device=self._device,**kwargs)
        if name=='eye':
            n=args[0]
            m=kwargs.pop('M',None)
            return t.eye(n,n if m is None else m,dtype=self._dtype(kwargs.get('dtype',self.real_dtype)),device=self._device)
        if name=='transpose':
            x,axes=args
            return x.permute(tuple(reversed(range(x.ndim))) if axes is None else tuple(axes))
        if name=='reshape':
            return args[0].reshape(args[1])
        if name=='moveaxis':
            return t.movedim(*args,**kwargs)
        if name=='imag' and not args[0].is_complex():
            return t.zeros_like(args[0])
        if name in ('sum','max','min','all','any'):
            axis=kwargs.pop('axis',None)
            keepdims=kwargs.pop('keepdims',False)
            if name=='sum':
                if kwargs.get('dtype') is not None:
                    kwargs['dtype']=self._dtype(kwargs['dtype'])
                return t.sum(args[0],dim=axis,keepdim=keepdims,**kwargs)
            if name in ('max','min'):
                reduce=t.amax if name=='max' else t.amin
                x=args[0]
                if x.is_complex():
                    real=reduce(x.real,dim=axis,keepdim=True)
                    infinity=-float('inf') if name=='max' else float('inf')
                    imag=reduce(t.where(x.real==real,x.imag,infinity),dim=axis,keepdim=True)
                    out=t.complex(real,imag)
                    if not keepdims:
                        if axis is None:
                            out=out.reshape(())
                        else:
                            axes=(axis,) if isinstance(axis,int) else axis
                            for index in sorted((a%x.ndim for a in axes),reverse=True):
                                out=out.squeeze(index)
                    return out
                return reduce(x,dim=axis,keepdim=keepdims)
            return getattr(t,name)(args[0],dim=axis,keepdim=keepdims)
        if name=='linalg.norm':
            kwargs['dim']=kwargs.pop('axis',None)
            kwargs['keepdim']=kwargs.pop('keepdims',False)
        if name.startswith('linalg.'):
            return getattr(t.linalg,name.split('.')[1])(*args,**kwargs)
        return getattr(t,name)(*args,**kwargs)

    def strict_update(self,name,x,idx,value):
        y=x.clone()
        if isinstance(idx,np.ndarray):
            values=self._torch.broadcast_to(value,idx.shape)
            for position,index in enumerate(idx):
                if name=='set':
                    y[int(index)]=values[position]
                elif name=='add':
                    y[int(index)]+=values[position]
                elif name=='sub':
                    y[int(index)]-=values[position]
                else:
                    y[int(index)]*=values[position]
        elif name=='set':
            y[idx]=value
        elif name=='add':
            y[idx]+=value
        elif name=='sub':
            y[idx]-=value
        else:
            y[idx]*=value
        return y

    def transpose(self,x,axes=None):
        return self.strict_call('transpose',x,axes)

    def tensordot(self,a,b,axes=2):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        if isinstance(axes,tuple):
            axes=tuple([int(v)] if isinstance(v,(int,np.integer)) else list(v) for v in axes)
        return self._torch.tensordot(a.to(dtype),b.to(dtype),dims=axes)

    def dot(self,a,b):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        return self._torch.matmul(a.to(dtype),b.to(dtype))

    def vdot(self,a,b):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        return self._torch.vdot(a.to(dtype).reshape(-1),b.to(dtype).reshape(-1))

    def maximum(self,a,b):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        return self._torch.maximum(a.to(dtype),b.to(dtype))

    def minimum(self,a,b):
        a,b=self.asarray(a),self.asarray(b)
        dtype=self._torch.promote_types(a.dtype,b.dtype)
        return self._torch.minimum(a.to(dtype),b.to(dtype))

    def concatenate(self,arrays,axis=0):
        return self._torch.cat(tuple(arrays),dim=axis)

    def ascontiguousarray(self,x,dtype=None):
        return self.asarray(x,dtype=dtype).contiguous()

    def iscomplexobj(self,x):
        return x.is_complex() if isinstance(x,self._torch.Tensor) else np.iscomplexobj(x)

    def __getattr__(self,name):
        # A bounded NumPy-style surface. Namespace availability does not imply
        # strict support; contexts advertise only tested operation contracts.
        if name in ('float32','float64','complex64','complex128','int32','int64','bool_'):
            return getattr(np,name)
        if name in ('zeros','ones','empty','full','eye','arange','sum','max','min','all','any',
                    'reshape','moveaxis','conj','real','imag','abs','sqrt','exp','log','sin','cos',
                    'isfinite','matmul','einsum','diag','diagonal','stack','hstack','vstack',
                    'allclose','isclose','where','sign','argsort','sort','clip'):
            return lambda *args,**kwargs:self.strict_call(name,*args,**kwargs)
        raise AttributeError(name)

    def sync(self):
        if self._device.type=='cuda':
            self._torch.cuda.synchronize(self._device)


class _TorchLinalg:
    def __init__(self,backend):
        self.backend=backend

    def __getattr__(self,name):
        if name not in ('qr','svd','eigh','eigvalsh','solve','norm','inv'):
            raise AttributeError(name)
        return lambda *args,**kwargs:self.backend.strict_call('linalg.'+name,*args,**kwargs)


class _TorchRandom:
    def __init__(self,backend):
        self.backend=backend

    def seed(self,value):
        self.backend._generator.manual_seed(value)

    @staticmethod
    def _shape(size):
        return () if size is None else ((int(size),) if isinstance(size,(int,np.integer)) else tuple(size))

    def uniform(self,low=0.,high=1.,size=None,dtype=None):
        b=self.backend
        return b._torch.rand(self._shape(size),dtype=b._dtype(dtype or b.real_dtype),device=b._device,generator=b._generator)*(high-low)+low

    def normal(self,loc=0.,scale=1.,size=None,dtype=None):
        b=self.backend
        return b._torch.randn(self._shape(size),dtype=b._dtype(dtype or b.real_dtype),device=b._device,generator=b._generator)*scale+loc

    def rand(self,*shape):
        return self.uniform(size=shape)

    def randn(self,*shape):
        return self.normal(size=shape)

    def random(self,size=None):
        return self.uniform(size=size)

    def randint(self,low,high=None,size=None,dtype=np.int64):
        if high is None:
            low,high=0,low
        b=self.backend
        return b._torch.randint(low,high,self._shape(size),dtype=b._dtype(dtype),device=b._device,generator=b._generator)
