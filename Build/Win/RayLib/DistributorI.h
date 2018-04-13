#pragma once

/**

Distributor Interface

Main interface for data distribution between nodes
Implementation has distributed system logics for creating cluster

It also is responsible for data transfer between nodes.
Distributer is not user interfaceable, and can only be used by Tracer, 
Visor and Analytic Classes/Programs.

Distributor is interfaces because it may have different implementations depending on the
distribution (symmetric/asymmetric computers, node types etc.)

Distributor has single thread per connection. Mostly 1-2 for neig
Other is for sending

*/

#include <cstdint>
#include <vector>

class TracerI;
struct RayStack;

class DistributorI
{
	public:
		enum NodeTypes
		{
			WORK_NODE,
			UI_NODE
		};

	private:
	protected:
		// Interface
		// Distributed Leader
		virtual void			EliminateNode() = 0;				// Leader removes node from the pool
		virtual void			IntroduceNode() = 0;				// Leader sends new node to pairs

		virtual void			StartFrame() = 0;					// Send Start Frame Command
		virtual void			RenderIntersect() = 0;				// Main intersect command
		virtual void			RenderGenRays() = 0;				// Main ray generation command
		virtual void			RenderEnd() = 0;					// Main rendering end command
		virtual void			AssignMaterial(uint32_t material,	// Assign material to node
											   uint32_t node) = 0;
		virtual void			PollNode(uint32_t) = 0;				// PollNode to check if its not dead

		// Distributed Non-Leader
		virtual void			RequestLeaderElection() = 0;	// Request a Leader election
		virtual void			RedirectCandidateNode() = 0;	// Redirect new node to leader

	public:
		virtual					~DistributorI() = default;

		// Interface
		// Tracer Related
		
		// Requesting (These are offline)
		virtual void			RequestObjectAccelerator() = 0;
		virtual void			RequestObjectAccelerator(uint32_t objId) = 0;
		virtual void			RequestObjectAccelerator(const std::vector<uint32_t>& objIds) = 0;

		virtual void			RequestScene() = 0;
		virtual void			RequestSceneMaterial(uint32_t) = 0;
		virtual void			RequestSceneObject(uint32_t) = 0;
		
		// Sending
		virtual void			SendMaterialRays(uint32_t materialId, const std::vector<RayStack>&) = 0;

		// Misc.
		virtual uint64_t		NodeId() = 0;
		virtual uint64_t		TotalMemory() = 0; // Returns entire node cluster's memory
};