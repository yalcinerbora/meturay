#pragma once
/**

Default type generation functions

Most of the time each DLL will came with its own
Class construction functions (which means calling constructor would suffice)

These functions define how generators should be defined. If a type will be generated
accross DLL boundaries it should declare

It also hints how a constructor should be defined. Accelerator should take primitive
as an input. (since those types are storngly tied)


*/
#include "RayLib/ObjectFuncDefinitions.h"
#include "RayLib/SceneStructs.h"

class GPUAcceleratorGroupI;
class GPUPrimitiveGroupI;

// Fundamental Pointers whichalso support creation-deletion
// across dll boundaries
using GPUAccelGPtr = SharedLibPtr<GPUAcceleratorGroupI>;
using GPUPrimGPtr = SharedLibPtr<GPUPrimitiveGroupI>;
using GPUMatGPtr = SharedLibPtr<GPUMaterialGroupI>;

using GPUAccelBPtr = SharedLibPtr<GPUAcceleratorBatchI>;
using GPUMatBPtr = SharedLibPtr<GPUMaterialBatchI>;

// Statically Inerfaced Generators
template<class Accel>
using AccelGroupGeneratorFunc = Accel* (&)(const GPUPrimitiveGroupI&,
										   const TransformStruct*);

template<class AccelBatch>
using AccelBatchGeneratorFunc = AccelBatch* (&)(const GPUAcceleratorGroupI&,
												const GPUPrimitiveGroupI&);

template<class MaterialBatch>
using MaterialBatchGeneratorFunc = MaterialBatch* (&)(const GPUMaterialGroupI&,
													  const GPUPrimitiveGroupI&);

//=========================//
// Shared Ptr Construction //
//=========================//
// Group
class GPUPrimGroupGen
{
	private:
		ObjGeneratorFunc<GPUPrimitiveGroupI>	gFunc;
		ObjDestroyerFunc<GPUPrimitiveGroupI>	dFunc;

	public:
		// Constructor & Destructor
		GPUPrimGroupGen(ObjGeneratorFunc<GPUPrimitiveGroupI> g,
						ObjDestroyerFunc<GPUPrimitiveGroupI> d)
			: gFunc(g)
			, dFunc(d) 
		{}

		GPUPrimGPtr operator()()
		{
			GPUPrimitiveGroupI* mat = gFunc();
			return GPUPrimGPtr(mat, dFunc);
		}
};

class GPUMatGroupGen
{
	private:
		ObjGeneratorFunc<GPUMaterialGroupI>	gFunc;
		ObjDestroyerFunc<GPUMaterialGroupI>	dFunc;

	public:
		// Constructor & Destructor
		GPUMatGroupGen(ObjGeneratorFunc<GPUMaterialGroupI> g,
					   ObjDestroyerFunc<GPUMaterialGroupI> d)
			: gFunc(g)
			, dFunc(d) 
		{}

		GPUMatGPtr operator()()
		{
			GPUMaterialGroupI* mat = gFunc();
			return GPUMatGPtr(mat, dFunc);
		}
};

class GPUAccelGroupGen
{
	private:
		AccelGroupGeneratorFunc<GPUAcceleratorGroupI>	gFunc;
		ObjDestroyerFunc<GPUAcceleratorGroupI>			dFunc;

	public:
		// Constructor & Destructor
		GPUAccelGroupGen(AccelGroupGeneratorFunc<GPUAcceleratorGroupI> g,
						 ObjDestroyerFunc<GPUAcceleratorGroupI> d)
			: gFunc(g)
			, dFunc(d) 
		{}

		GPUAccelGPtr operator()(const GPUPrimitiveGroupI& pg,
								const TransformStruct* ts)
		{
			GPUAcceleratorGroupI* mat = gFunc(pg, ts);
			return GPUAccelGPtr(mat, dFunc);
		}
};
// Batch
class GPUAccelBatchGen
{
	private:
		AccelBatchGeneratorFunc<GPUAcceleratorBatchI>	gFunc;
		ObjDestroyerFunc<GPUAcceleratorBatchI>			dFunc;

	public:
		// Constructor & Destructor
		GPUAccelBatchGen(AccelBatchGeneratorFunc<GPUAcceleratorBatchI> g,
						 ObjDestroyerFunc<GPUAcceleratorBatchI> d)
			: gFunc(g)
			, dFunc(d) 
		{}

		GPUAccelBPtr operator()(const GPUAcceleratorGroupI& ag,
								const GPUPrimitiveGroupI& pg)
		{
			GPUAcceleratorBatchI* accel = gFunc(ag, pg);
			return GPUAccelBPtr(accel, dFunc);
		}
};

class GPUMatBatchGen
{
	private:
		MaterialBatchGeneratorFunc<GPUMaterialBatchI>	gFunc;
		ObjDestroyerFunc<GPUMaterialBatchI>				dFunc;

	public:
		// Constructor & Destructor
		GPUMatBatchGen(MaterialBatchGeneratorFunc<GPUMaterialBatchI> g,
					   ObjDestroyerFunc<GPUMaterialBatchI> d)
			: gFunc(g)
			, dFunc(d)
		{}

		GPUMatBPtr operator()(const GPUMaterialGroupI& mg,
							  const GPUPrimitiveGroupI& pg)
		{
			GPUMaterialBatchI* accel = gFunc(mg, pg);
			return GPUMatBPtr(accel, dFunc);
		}
};

namespace TypeGenWrappers
{
	//==============//
	// New Wrappers //
	//==============//
	template <class Base, class T>
	Base* DefaultConstruct()
	{
		return new T();
	}

	template <class T>
	void DefaultDestruct(T* t)
	{
		return delete t;
	}

	template <class T>
	void EmptyDestruct(T* t) {}

	template <class Base, class AccelGroup>
	Base* AccelGroupConstruct(const GPUPrimitiveGroupI& p,
							  const TransformStruct* t)
	{
		return new AccelGroup(p, t);
	}

	template <class Base, class AccelBatch>
	Base* AccelBatchConstruct(const GPUAcceleratorGroupI& a,
							  const GPUPrimitiveGroupI& p)
	{
		return new AccelBatch(a, p);
	}

	template <class Base, class MatBatch>
	Base* MaterialBatchConstruct(const GPUMaterialGroupI& m,
								 const GPUPrimitiveGroupI& p)
	{
		return new MatBatch(m, p);
	}	
}