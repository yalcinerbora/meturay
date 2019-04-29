#pragma once
/**

Static Interface of a BVH traversal
with custom Intersection and Hit acceptance

*/

#include "AcceleratorDeviceFunctions.h"
#include "RayLib/HitStructs.h"
//
//// Fundamental BVH Tree Node
//template<class LeafStruct>
//struct alignas(8) BVHNode
//{
//  static constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();
//
//  // Pointers
//  union
//  {
//      struct
//      {
//          // 8 Word
//          Vector3 aabbMin;
//          uint32_t left;
//          Vector3 aabbMax;
//          uint32_t right;
//          // 1 Word
//          uint32_t parent;
//      };
//      LeafStruct leaf;
//  };
//  bool isLeaf;
//};
//
//
//// This is fundemental BVH traversal kernel
//// It supparts partial traversal and continuation traversal(for scene tree)
//// It is customizable by intersection and hit determination
//// Its output is customizable by HitStructs
//template <class HitGMem, class HitReg,
//        class LeafStruct, class PrimitiveData,
//        IntersctionFunc<LeafStruct, PrimitiveData> IFunc,
//        AcceptHitFunc<HitReg> AFunc>
//__global__ void KCIntersectBVH(// I-O
//                             RayGMem* gRays,
//                             HitGMem* gHitStructs,
//                             HitKey*  gCurrentHits,
//                             // Input
//                             const RayId* gRayIds,
//                             const HitKey* gPotentialHits,
//                             const uint32_t rayCount,
//                             // Constants
//                             const BVHNode<LeafStruct>** gBVHList,
//                             const Matrix4x4* gInverseTransforms,
//                             const PrimitiveData gPrimData)
//{
//  // Convenience Functions
//  auto IsAlreadyTraversed = [](uint64_t list, uint32_t depth) -> bool
//  {
//      return ((list >> depth) & 0x1) == 1;
//  };
//  auto MarkAsTraversed(uint64_t& list, uint32_t depth)
//  {
//      list += (1 << depth);
//  };
//  auto Pop = [](uint64_t& list, uint32_t& depth)
//  {
//      MarkAsTraversed(list, depth);
//      depth++;
//  };
//
//  // Grid Stride Loop
//  for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
//      globalId < rayCount; globalId += blockDim.x * gridDim.x)
//  {
//      const RayId id = gRayIds[globalId];
//      const HitKey key = gHitKeys[id];
//
//      // Key is the index of the inner BVH
//      const BVHNode<LeafStruct>* gBVH = gBVHList[key];
//
//      // Load Ray/Hit to Register
//      RayReg ray(gRays, id);
//      HitReg hit(gHits, id);
//
//      // Transform the ray to the local space
//      // If applicable
//      if(gInverseTransforms != nullptr)
//      {
//          const Matrix4x4 t = gInverseTransforms[key];
//          ray.ray.TransformSelf(t);
//      }
//
//      // Depth First Search over BVH
//      const uint32_t depth = sizeof(uint64_t) * 8;
//      BVHNode<LeafStruct> currentNode = gBVH[0];
//      for(uint64_t& list = 0; list < 0xFFFFFFFF;)
//      {
//          // Fast pop if both of the children is carries current node is zero
//          // (This means that bit is carried)
//          if(IsAlreadyTraversed(list, depth))
//          {
//              currentNode = gBVH[currentNode.parent];
//              Pop(list, depth);
//          }
//          // Check if we already traversed left child
//          // If child bit is on this means lower left child is traversed
//          else if(IsAlreadyTraversed(list, depth - 1) &&
//                  currentNode.right != BVHNode<LeafStruct>::NULL_NODE)
//          {
//              // Go to right child
//              currentNode = gBVH[currentNode.right];
//              depth--;
//          }
//          // Now this means that we entered to this node first time
//          // Check if this node is leaf or internal
//          // Check if it is leaf node
//          else if(currentNode.isLeaf)
//          {
//              // Do Intersection Test
//              float newT = IFunc(ray, node.leaf, primitiveList);
//              // Do Hit Acceptance break traversal if terminate is called
//              if(AFunc(hit, ray, newT)) break;
//
//              // Continue
//              Pop(list, depth);
//          }
//          // Not leaf so check AABB
//          else if(ray.ray.IntersectsAABB(currentNode.aabbMin, currentNode.aabbMax)
//          {
//              // Go left if avail
//              if(currentNode.left != BVHNode<LeafStruct>::NULL_NODE)
//              {
//                  currentNode = gBVH[currentNode.left];
//                  depth--;
//              }
//              // If not avail and since we are first time on this node
//              // Try to go right
//              else if(currentNode.left != BVHNode<LeafStruct>::NULL_NODE)
//              {
//                  // In this case dont forget to mark left child as traversed
//                  MarkAsTraversed(list, depth - 1);
//
//                  currentNode = gBVH[currentNode.right];
//                  depth--;
//              }
//              else
//              {
//                  // This should not happen
//                  // since we have "isNode" boolean
//                  assert(false);
//
//                  // Well in order to be correctly mark this node traversed also
//                  // In the next iteration node will pop itself
//                  MarkAsTraversed(list, depth - 1);
//              }
//          }
//          // Finally no ray is intersected
//          // Go to parent
//          else
//          {
//              // Skip Leafs
//              currentNode = gBVH[currentNode.parent];
//              Pop(list, depth);
//          }
//      }
//      // Write Updated Stuff
//      // Only tMax of ray which could have changed
//      ray.UpdateTMax(rays, globalId);
//      hit.Update(hits, globalId);
//  }
//}
//
//__global__ void KCIntersectBaseBVH(// I-O
//                                 HitKey* gHitKeys,
//                                 uint32_t* gPrevNode,
//                                 uint64_t* gPrevList,
//                                 // Input
//                                 const RayGMem* gRays,
//                                 const RayId* gRayIds,
//                                 const uint32_t rayCount,
//                                 // Constants
//                                 const BVHNode<BaseLeaf>* gBVH)
//{
//  // Convenience Functions
//  auto IsAlreadyTraversed = [](uint64_t list, uint32_t depth) -> bool
//  {
//      return ((list >> depth) & 0x1) == 1;
//  };
//  auto MarkAsTraversed(uint64_t& list, uint32_t depth)
//  {
//      list += (1 << depth);
//  };
//  auto Pop = [](uint64_t& list, uint32_t& depth)
//  {
//      MarkAsTraversed(list, depth);
//      depth++;
//  };
//
//  // Grid Stride Loop
//  for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
//      globalId < rayCount; globalId += blockDim.x * gridDim.x)
//  {
//      const RayId id = gRayIds[globalId];
//
//      // Load initial traverse point if available
//      uint32_t initalLoc = gPrevNode[id];
//      uint64_t list = gPrevList[id];
//
//      // Load Ray/Hit to Register
//      RayReg ray(gRays, id);
//      HitKey key = gHitKeys[globalId];
//      if(key == HitConstants::InvalidKey) continue;
//
//      // Depth First Search over BVH
//      uint32_t depth = sizeof(uint64_t) * 8;
//      BVHNode<BaseLeaf> currentNode = gBVH[initalLoc];
//      while(list < 0xFFFFFFFF)
//      {
//          // Fast pop if both of the children is carries current node is zero
//          // (This means that bit is carried)
//          if(IsAlreadyTraversed(list, depth))
//          {
//              currentNode = gBVH[currentNode.parent];
//              initalLoc = currentNode.parent;
//              Pop(list, depth);
//          }
//          // Check if we already traversed left child
//          // If child bit is on this means lower left child is traversed
//          else if(IsAlreadyTraversed(list, depth - 1) &&
//                  currentNode.right != BVHNode<BaseLeaf>::NULL_NODE)
//          {
//              // Go to right child
//              currentNode = gBVH[currentNode.right];
//              initalLoc = currentNode.right;
//              depth--;
//          }
//          // Now this means that we entered to this node first time
//          // Check if this node is leaf or internal
//          // Check if it is leaf node
//          else if(currentNode.isLeaf)
//          {
//              key = currentNode.leaf.key;
//              break;
//          }
//          // Not leaf so check AABB
//          else if(ray.ray.IntersectsAABB(currentNode.aabbMin, currentNode.aabbMax))
//          {
//              // Go left if avail
//              if(currentNode.left != BVHNode<BaseLeaf>::NULL_NODE)
//              {
//                  currentNode = gBVH[currentNode.left];
//                  initalLoc = currentNode.left;
//                  depth--;
//              }
//              // If not avail and since we are first time on this node
//              // Try to go right
//              else if(currentNode.left != BVHNode<BaseLeaf>::NULL_NODE)
//              {
//                  // In this case dont forget to mark left child as traversed
//                  MarkAsTraversed(list, depth - 1);
//
//                  currentNode = gBVH[currentNode.right];
//                  initalLoc = currentNode.left;
//                  depth--;
//              }
//              else
//              {
//                  // This should not happen
//                  // since we have "isNode" boolean
//                  assert(false);
//                  // Well in order to be correctly mark this node traversed also
//                  // In the next iteration node will pop itself
//                  MarkAsTraversed(list, depth - 1);
//              }
//          }
//          // Finally no ray is intersected
//          // Go to parent
//          else
//          {
//              // Skip Leafs
//              currentNode = gBVH[currentNode.parent];
//              initalLoc = currentNode.parent;
//              Pop(list, depth);
//          }
//      }
//      // Write Updated Stuff
//      gPrevNode[id] = initalLoc;
//      gPrevList[id] = list;
//      gHitKeys[globalId] = key;
//  }
//}
//
//
//// These are fundamental BVH generation Kernels
//// These can be implemented by custom aabb fetch functions
//
//// Writes surface area of each primitive
//// These will be reduced to an average value.
//template <class PrimitiveData,
//        AreaGenFunc<PrimitiveData> AreaFunc>
//__global__ void KCGeneratePrimitiveAreaBVH(float* gOutArea,
//                                         // Input
//                                         const PrimitiveData gPrimData,
//                                         const uint32_t primtiveCount)
//{
//  // Grid Stride Loop
//  for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
//      globalId < primtiveCount; i += blockDim.x * gridDim.x)
//  {
//      float area = AreaFunc(globalId, gPrimData);
//      gOutArea[globalId] = area;
//  }
//}
//
//// Determines how many cells should a primitive cover
//template <class PrimitiveData,
//        BoxGenFunc<PrimitiveData> BoxFunc,
//        AreaGenFunc<PrimitiveData> AreaFunc>
//__global__ void KCDetermineCellCountBVH(uint32_t* gOutCount,
//                                      // Input
//                                      const PrimitiveData gPrimData,
//                                      const uint32_t primtiveCount,
//                                      const float optimalArea)
//{
//  // Grid Stride Loop
//  for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
//      globalId < primtiveCount; i += blockDim.x * gridDim.x)
//  {
//      // Generate your data
//      AABB3f aabb = BoxFunc(globalId, gPrimData);
//      float primitiveArea = AreaGenFunc(globalId, gPrimData);
//
//      // Compare yoursef with area to generate triangles
//      // Find how many splits are on you
//      // Get limitation
//      uint32_t splitCount;
//
//      // TODO: implement
//
//      gOutCount[globalId] = cellCount;
//  }
//}
//
//// Here do scan over primitive count for
//
//// Generates Partial AABB and morton numbers for each partial data
//template <class PrimitiveData, BoxGenFunc<PrimitiveData> BoxFunc>
//__global__ void KCGenerateParitalDataBVH(AABB3f* gSubAABBs,
//                                       uint64_t* gMortonCodes,
//                                       // Input
//                                       const uint32_t* gPrimId,
//                                       const PrimitiveData gPrimData,
//                                       const uint32_t subPrimtiveCount,
//                                       const AABB3f& extents,
//                                       const float optimalArea)
//{
//  // Grid Stride Loop
//  for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
//      globalId < subPrimtiveCount; i += blockDim.x * gridDim.x)
//  {
//      // Generate parent data
//      uint32_t parentId = gPrimId[globalId];
//      AABB3f aabb = BoxFunc(parentId, gPrimData);
//      float primitiveArea = AreaGenFunc(parentId, gPrimData);
//
//      // Using parent primitive data and relative index
//      // Find subAABB
//      // And generate Morton code for that AABB centroid
//      AABB3f subAABB = aabb; // TODO: calculate
//      uint64_t morton = DiscretizePointMorton(subAABB.Centroid(),
//                                              extents, optimalArea);
//
//      gSubAABBs[globalId] = subAABB;
//      gMortonCodes[globalId] = morton;
//  }
//}
//
//// After that do a sort over morton codes
//template <class LeafStruct, class PrimitiveData>
//__global__ void KCGenerateBVH(BVHNode<LeafStruct>* gBVHList,
//                            //
//                            const uint32_t* gPrimId,
//                            const uint64_t* gMortonCodes,
//
//                            const uint32_t subPrimtiveCount)
//{
//  uint32_t internalNodeCount = subPrimtiveCount - 1;
//
//  // Grid Stride Loop
//  for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
//      globalId < internalNodeCount; i += blockDim.x * gridDim.x)
//  {
//      // Binary Search
//
//
//  }
//}