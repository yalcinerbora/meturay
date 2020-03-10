#include "MDistributor.h"

//// Interface
//// Distributed Leader
//void MDistributor::EliminateNode()
//{
//  assert(isLeader);
//  //TODO
//}
//
//void MDistributor::IntroduceNode()
//{
//  assert(isLeader);
//  //TODO
//}
//
//void MDistributor::StartFrame()
//{
//  assert(isLeader);
//  //TODO
//}
//
//void MDistributor::RenderIntersect()
//{
//  assert(isLeader);
//  //TODO
//}
//
//void MDistributor::RenderGenRays()
//{
//  assert(isLeader);
//  //TODO
//}
//
//void MDistributor::RenderEnd()
//{
//  assert(isLeader);
//  //TODO
//}
//
//void MDistributor::AssignMaterial(uint32_t material,
//                                uint32_t node)
//{
//  assert(isLeader);
//  //TODO
//}
//void MDistributor::PollNode(uint32_t)
//{
//  assert(isLeader);
//  //TODO
//}
//
//void MDistributor::RequestLeaderElection()
//{
//  //TODO
//}
//
//void MDistributor::RedirectCandidateNode()
//{
//  //TODO
//}
//
//MDistributor::MDistributor()
//{}
//
//bool MDistributor::Alone()
//{
//  //TODO:
//}
//
//bool MDistributor::CheckIfRenderRequested(uint32_t renderCount)
//{
//
//}
//
//void MDistributor::SendMaterialRays(uint32_t materialId,
//                                  const std::vector<RayStack>)
//{
//
//}
//
//void MDistributor::SendMaterialRays(const std::vector<ArrayPortion<uint32_t>> materialIds,
//                                  const std::vector<RayStack>)
//{
//
//}
//
//void MDistributor::SendImage(const std::vector<Vector3f> image,
//                           const Vector2ui resolution,
//                           const Vector2ui offset,
//                           const Vector2ui size)
//{
//  // PLACEHOLDER
//  if(displayCallback)
//      displayCallback(std::move(image), resolution, size, offset);
//}
//
//void MDistributor::RequestObjectAccelerator()
//{
//
//}
//
//void MDistributor::RequestObjectAccelerator(uint32_t objId)
//{
//
//}
//
//void MDistributor::RequestObjectAccelerator(const std::vector<uint32_t>& objIds)
//{
//
//}
//
//void MDistributor::RequestScene()
//{
//
//}
//
//void MDistributor::RequestSceneMaterial(uint32_t)
//{
//
//}
//
//void MDistributor::RequestSceneObject(uint32_t)
//{
//
//}
//
//void MDistributor::RequestMaterialRays(const std::vector<RayStack>&)
//{
//
//}
//
//uint64_t MDistributor::NodeId()
//{
//
//}
//
//uint64_t MDistributor::TotalCPUMemory()
//{
//
//}
//
//uint64_t MDistributor::TotalGPUMemory()
//{
//
//}
//
//void MDistributor::SetImageStream(bool)
//{
//
//}
//
//void MDistributor::SetImagePeriod(double seconds)
//{
//
//}
//
//void  MDistributor::ChangeCamera(const CPUCamera&)
//{
//
//}
//
//void  MDistributor::ChangeTime(double seconds)
//{
//
//}
//
//void  MDistributor::ChangeParameters()
//{
//
//}
//
//void  MDistributor::ChangeFPS(int fps)
//{
//
//}
//
//void  MDistributor::NextFrame()
//{
//
//}
//
//void  MDistributor::PreviousFrame()
//{
//
//}
//
//void  MDistributor::AttachDisplayCallback(SetImageSegmentFunc)
//{
//
//}