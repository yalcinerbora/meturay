#pragma once


/**

CUDA Device Memory RAII principle classes

New unified memory classes are used where applicable
These are wrapper of cuda functions and their most important responsiblity is 
to delete allocated memory

All of the operations (execpt allocation) are asyncronious.

TODO: should we interface these?

*/

// Basic semi-interface for memories that are static for each GPU
// Textures are one example
class DeviceLocalMemoryI
{
	protected:
		int						currentDevice;

	public:
								DeviceLocalMemoryI(int initalDevice = 0) : currentDevice(initalDevice) {}
		virtual					~DeviceLocalMemoryI() = default;

		// Interface
		virtual void			MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) = 0;
};

// Has a CPU Image of current memory
// Usefull for device static memory that can be generated at CPU while 
// GPU doing work on GPU memory
// in our case some form of function backed animation can be calculated using these)
class DeviceMemoryCPUBacked : public DeviceLocalMemoryI
{
	private:		
		void*						h_ptr;
		void*						d_ptr;

	protected:
	public:
		// Constructors & Destructor
									DeviceMemoryCPUBacked() = delete;
									DeviceMemoryCPUBacked(size_t sizeInBytes);
									DeviceMemoryCPUBacked(const DeviceMemoryCPUBacked&);
									DeviceMemoryCPUBacked(DeviceMemoryCPUBacked&&);
									~DeviceMemoryCPUBacked();
		DeviceMemoryCPUBacked&		operator=(const DeviceMemoryCPUBacked&);
		DeviceMemoryCPUBacked&		operator=(DeviceMemoryCPUBacked&&);
	
		// Memcopy
		void						CopyToDevice(cudaStream_t stream = nullptr);
		void						CopyToHost(cudaStream_t stream = nullptr);
};

// Generic Device Memory (most of the cases this should be used)
// Fire and forget type memory
// In our case rays and hit records will be stored in this form
class DeviceMemory
{
	private:
		void*						m_ptr;	// managed pointer

	protected:
	public:
		// Constructors & Destructor
									DeviceMemory() = delete;
									DeviceMemory(size_t sizeInBytes);
									DeviceMemory(const DeviceMemory&);
									DeviceMemory(DeviceMemory&&);
									~DeviceMemory();
		DeviceMemory&				operator=(const DeviceMemory&);
		DeviceMemory&				operator=(DeviceMemory&&);

		// Access
		template<class T>
		constexpr explicit			operator T*();
		template<class T> 
		constexpr explicit			operator const T*() const;
		constexpr 					operator void*();
		constexpr 					operator const void*() const;
};