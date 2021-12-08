//#pragma once
///**
//
//Distributed version of MDistributor
//
//Each process will  have single Distributor and a Tracer or Visor
//
//Visor is responsible for delegating commands to multiple Tracer Nodes
//throughout this interface (using udp or tcp we will see)
//
//*/
//
//#include "DistributorI.h"
//#include <asio.hpp>
//#include <deque>
//
//using asio::ip::tcp;
//
//class MDistributor : public DistributorI
//{
//
//  public:
//      enum Type
//      {
//          TRACER,
//          VISOR
//      };
//
//      struct Node
//      {
//          uint32_t id;
//          asio::ip::address address;
//
//
//      };
//
//
//  private:
//      bool                    isLeader;
//      std::deque<Node>        nodes;
//      Node*                   leader;
//
//      // Visor Callback (If this distributor is visor distributor)
//      SetImageSegmentFunc     displayCallback;
//
//      // ...
//
//
//  protected:
//      // Interface
//      // Distributed Leader
//      void                    EliminateNode() override;               // Leader removes node from the pool
//      void                    IntroduceNode() override;               // Leader sends new node to pairs
//
//      void                    StartFrame() override;                  // Send Start Frame Command
//      void                    RenderIntersect() override;             // Main intersect command
//      void                    RenderGenRays() override;               // Main ray generation command
//      void                    RenderEnd() override;                   // Main rendering end command
//      void                    AssignMaterial(uint32_t material,       // Assign material to node
//                                             uint32_t node) override;
//      void                    PollNode(uint32_t) override;            // PollNode to check if its not dead
//
//      // Distributed Non-Leader
//      void                    RequestLeaderElection() override;       // Request a Leader election
//      void                    RedirectCandidateNode() override;       // Redirect new node to leader
//
//  public:
//      // Constructors & Destructor
//                              MDistributor();
//                              ~MDistributor() = default;
//
//      // ================= //
//      //  Tracer Related   //
//      // ================= //
//      // Sending (All non-blocking)
//      void                    SendMaterialRays(uint32_t materialId,
//                                               const std::vector<RayStack>) override;
//      void                    SendMaterialRays(const std::vector<ArrayPortion<uint32_t>> materialIds,
//                                               const std::vector<RayStack>) override;
//      void                    SendImage(const std::vector<Vector3f> image,
//                                        const Vector2ui resolution,
//                                        const Vector2ui offset = Vector2ui(0, 0),
//                                        const Vector2ui size = Vector2ui(0, 0)) override;
//
//      // Requesting (All are blocking)
//      void                    RequestObjectAccelerator() override;
//      void                    RequestObjectAccelerator(uint32_t objId) override;
//      void                    RequestObjectAccelerator(const std::vector<uint32_t>& objIds) override;
//
//      void                    RequestScene() override;
//      void                    RequestSceneMaterial(uint32_t) override;
//      void                    RequestSceneObject(uint32_t) override;
//
//      // Request rays that are responsible by this node
//      void                    RequestMaterialRays(const std::vector<RayStack>&) override;
//
//      // Misc.
//      uint64_t                NodeId() override;
//      uint64_t                TotalCPUMemory() override;
//      uint64_t                TotalGPUMemory() override;
//
//      // Check if render is requested for this frame
//      bool                    CheckIfRenderRequested(uint32_t renderCount) override;
//      // Check if distributed system is distributed at all
//      bool                    Alone() override;
//
//      // Receiving data from callbacks
//      void                    AttachCameraCallback(SetCameraFunc) override;
//      void                    AttachTimeCallback(SetTimeFunc) override;
//      void                    AttachParamCallback(SetParameterFunc) override;
//      void                    AttachFPSCallback(SetFPSFunc) override;
//      void                    AttachFrameCallback(SetFrameCallback) override;
//
//      // ================= //
//      //   Visor Related   //
//      // ================= //
//      void                    SetImageStream(bool) override;
//      void                    SetImagePeriod(double seconds) override;
//
//      // Visor Window I-O
//      void                    ChangeCamera(const CPUCamera&) override;
//      void                    ChangeTime(double seconds) override;
//      void                    ChangeParameters() override;
//      void                    ChangeFPS(int fps) override;
//      void                    NextFrame() override;
//      void                    PreviousFrame() override;
//
//      void                    AttachDisplayCallback(SetImageSegmentFunc) override;
//
//      // Visor CLI
//      //...
//};
