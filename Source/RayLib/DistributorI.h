#pragma once

/**

Tracer Distributor Interface

Main interface for data distribution between nodes
Implementation has distributed system logics for creating cluster

It also is responsible for data transfer between nodes. 
Distributor is main backend for all nodes (Analytic / Visor, Tracer)

*/

#include <cstdint>
#include <vector>

#include "ArrayPortion.h"
#include "Vector.h"

namespace asio::ip
{
	class adress;
}

enum class DistError
{
	OK,
};

class DistributorI
{
	public:
		enum CommandType
		{
			REQUEST,
			RECIEVE
		};

		enum CommandTag
		{
			ACCELERATOR,
			SCENE,
			MATERIAL,
			OBJECT,

			// Most important tag (used in per-bounce basis)
			MAT_RAY,
		};

		// Main Recieve
		typedef void(*RecieveFunc)(CommandTag, std::vector<const char*>);

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
		virtual void			RequestLeaderElection() = 0;		// Request a Leader election
		virtual void			RedirectCandidateNode() = 0;		// Redirect new node to leader

	public:
		virtual					~DistributorI() = default;

		// Main Interface
		virtual DistError		Connect(asio::ip::adress, int port) = 0;

		// Sending Data (non-block)
		virtual void			Request(const CommandTag) = 0;
		virtual void			Send(const CommandTag, 
									 const std::vector<const char*>) = 0;

		// Recieving Data
		virtual	void			AttachRecieveCallback(RecieveFunc) = 0;

		// Misc
		virtual uint64_t		NodeId() = 0;
	};