//#include "SceneGPU.h"
////#include "RayLib/RayHitStructs.h"
//#include "RayLib/SceneIO.h"
//#include "RayLib/Error.h"
//
//SceneGPU::SceneGPU(const SceneFile& scene)
//{
//	// TODO: Implement All
//
//	// Volumes	
//	for(const auto& vol : scene.volumes)
//	{
//		switch(vol.type)
//		{
//			case VolumeType::MAYA_NCACHE_FLUID:
//			{
//				// Volume				
//				ncVolumes.emplace_back(vol.fileName, vol.materialId, vol.surfaceId);
//				volumes.push_back(ncVolumes.back());
//				
//				ncVolumes.back().Load();
//
//				break;
//			}
//			default:
//				break;
//		}
//	}
//	// Materials
//	for(const auto& fMat : scene.fluidMaterials)
//	{
//		fMaterials.emplace_back(fMat.materialId,
//								fMat.ior,
//								fMat.colors,
//								fMat.colorInterp,
//								fMat.opacities,
//								fMat.opacityInterp,
//								fMat.transparency,
//								fMat.absorbtionCoeff,
//								fMat.scatteringCoeff);
//		fMaterials.back().Load();
//		materials.push_back(fMaterials.back());
//	}
//
//}
//	
//const MaterialI& SceneGPU::Material(uint32_t id) const
//{
//	return materials[id];
//}
//
//const MaterialList& SceneGPU::Materials() const
//{
//	return materials;
//}
//
//const SurfaceI& SceneGPU::Surface(uint32_t id) const
//{
//	return surfaces[id];
//}
//
//const SurfaceList& SceneGPU::Surfaces() const
//{
//	return surfaces;
//}
//
//AnimatableList& SceneGPU::Animatables()
//{
//	return animatables;
//}
//
//const MeshBatchI& SceneGPU::MeshBatch(uint32_t id)
//{
//	return  batches[id];
//}
//
//const MeshBatchList& SceneGPU::MeshBatch()
//{
//	return batches;
//}
//
//VolumeI& SceneGPU::Volume(uint32_t id)
//{
//	return volumes[id];
//}
//
//VolumeList& SceneGPU::Volumes()
//{
//	return volumes;
//}
//
//const LightI& SceneGPU::Light(uint32_t id)
//{
//	return lights[id];
//}
//
//void SceneGPU::ChangeTime(double timeSec)
//{
//}
////
////void SceneGPU::HitRays(uint32_t* location,
////					   const ConstRayRecordGMem,
////					   uint32_t rayCount) const
////{
////
////}
//
//
//#define _USE_MATH_DEFINES
//#define erand48(dummy) (double(rand()) / RAND_MAX)
//
//
//
//
//#include <math.h> // smallpt, a Path Tracer by Kevin Beason, 2008
//#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
//#include <stdio.h>
//
//#include <time.h>
//clock_t t0, t1;
//
//
//typedef struct Vec
//{ // Usage: time ./explicit 16 && xv image.ppm
//	double x, y, z; // position, also color (r,g,b)
//	Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }
//	Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
//	Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
//	Vec operator*(double b) const { return Vec(x*b, y*b, z*b); }
//	Vec mult(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
//	Vec& norm() { return *this = *this * (1 / sqrt(x*x + y*y + z*z)); }
//	double dot(const Vec &b) const { return x*b.x + y*b.y + z*b.z; } // cross:
//	Vec operator%(Vec&b) { return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x); }
//} const cVec;
//struct Ray { Vec o, d; Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };
//enum Refl_t { DIFF, SPEC, REFR }; // material types, used in radiance()
//struct Sphere
//{
//	double rad, rad2; // radius
//	Vec p, e, c; // position, emission, color
//	Refl_t refl; // reflection type (DIFFuse, SPECular, REFRactive)
//	Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
//		rad(rad_), p(p_), e(e_), c(c_), refl(refl_)
//	{
//		rad2 = rad_*rad_;
//	}
//	double intersect(const Ray &r) const
//	{ // returns distance, 0 if nohit
//		Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
//		double t, eps = 1e-4, b = op.dot(r.d), det = b*b - op.dot(op) + rad2;
//		if(det<0) return 0; else det = sqrt(det);
//		return (t = b - det)>eps ? t : ((t = b + det)>eps ? t : 0);
//	}
//};
//
//
//#define PI2 (M_PI + M_PI)
//
//
//Sphere spheres[] = {//Scene: radius, position, emission, color, material
//	Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left
//	Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght
//	Sphere(1e5, Vec(50,40.8, 1e5), Vec(),Vec(.75,.75,.75),DIFF),//Back
//	Sphere(1e5, Vec(50,40.8,-1e5 + 170), Vec(),Vec(), DIFF),//Frnt
//	Sphere(1e5, Vec(50, 1e5, 81.6), Vec(),Vec(.75,.75,.75),DIFF),//Botm
//	Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top
//	Sphere(16.5,Vec(27,16.5,47), Vec(),Vec(1,1,1)*.999, SPEC),//Mirr
//	Sphere(16.5,Vec(73,16.5,78), Vec(),Vec(1,1,1)*.999, REFR),//Glas
//															  //
//															  Sphere(5e-3,Vec(50,81.6 - 36.5,81.6),Vec(4,4,4)*1e7, Vec(), DIFF),//Lite
//};
//
//
///*
//Sphere spheres[] = {//Scene: radius, position, emission, color, material
//Sphere(1e5, Vec( 1e5+1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left
//Sphere(1e5, Vec(-1e5+99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght
//Sphere(1e5, Vec(50,40.8, 1e5), Vec(),Vec(.75,.75,.75),DIFF),//Back
//Sphere(1e5, Vec(50,40.8,-1e5+170), Vec(),Vec(), DIFF),//Frnt
//Sphere(1e5, Vec(50, 1e5, 81.6), Vec(),Vec(.75,.75,.75),DIFF),//Botm
//Sphere(1e5, Vec(50,-1e5+81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top
//Sphere(16.5,Vec(27,16.5,47), Vec(),Vec(1,1,1)*.999, SPEC),//Mirr
//Sphere(16.5,Vec(73,16.5,78), Vec(),Vec(1,1,1)*.999, REFR),//Glas
//Sphere(600, Vec(50,681.6-.27,81.6),Vec(12,12,12), Vec(), DIFF) //Lite
//};
//*/
//
///*
////from explicit.cpp
//Sphere spheres[] = {//Scene: radius, position, emission, color, material
//Sphere(1e5, Vec( 1e5+1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left
//Sphere(1e5, Vec(-1e5+99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght
//Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back
//Sphere(1e5, Vec(50,40.8,-1e5+170), Vec(),Vec(),           DIFF),//Frnt
//Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm
//Sphere(1e5, Vec(50,-1e5+81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top
//Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1)*.999, SPEC),//Mirr
//Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1)*.999, REFR),//Glas
//Sphere(1.5, Vec(50,81.6-16.5,81.6),Vec(4,4,4)*100,  Vec(), DIFF),//Lite
//};
//*/
//
//
////	3/ 1* 2* 1* 2/ 2dot   => 2/ 1* 1* 1dot 1* 1/
////****** start
//double molif_r = 1.; // Global mollification radius, shrinks per sample
//double molif_r2 = 1.; // Global mollification radius, shrinks per sample
//
//double mollify(Vec& l, const Vec rd, Vec& n, Vec& nl, double dist, int type)
//{
//
//	double cos_max = 1. / sqrt(1. + (molif_r2 / (dist*dist)));// Cone angle
//
//
//	Vec out;
//
//	if(type == REFR)
//	{
//
//		bool bigger = n.dot(nl)>0;
//
//		// Compute refraction vector
//		double nc = 1, nt = 1.5, nnt = bigger ? nc / nt : nt / nc, ddn = rd.dot(nl), cos2t;
//
//		if((cos2t = 1 - nnt*nnt*(1 - ddn*ddn))>0) // Refraction vector
//		{
//			out = (rd*nnt - n*(
//				//(n.dot(nl)>0?1:-1)*(ddn*nnt+sqrt(cos2t))
//				(bigger
//				 ?
//				 (ddn*nnt + sqrt(cos2t)) : -(ddn*nnt + sqrt(cos2t))
//				 )
//				)
//				).norm();
//		}
//	}
//	else
//	{
//		out = rd - (n + n)*n.dot(rd); // Reflection vector
//	}
//
//	/*
//	double solid_angle=PI2*(1.-cos_max); // Solid angle of the cone
//	return l.dot(out)>=cos_max ? (1./(solid_angle*l.dot(out))):0.; // Mollify
//	*/
//
//	//double solid_angle=PI2*(1.-cos_max); // Solid angle of the cone
//	return l.dot(out) >= cos_max ? (1. / (PI2*(1. - cos_max)*l.dot(out))) : 0.; // Mollify
//}
////****** end