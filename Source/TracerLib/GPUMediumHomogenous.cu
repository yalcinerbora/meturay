#include "GPUMediumHomogenous.cuh"

SceneError CPUMediumHomogenous::InitializeGroup(const NodeListing& mediumNodes,
												double time,
												const std::string& scenePath)
{

    //std::vector<GPUMediumHomogenous>
    std::vector<Vector3> absorbtionList;
    std::vector<Vector3> scaterringList;
    std::vector<float> iorList;
    std::vector<float> phaseList;

    for(const auto& node : mediumNodes)
    {
        std::vector<Vector3> nodeAbsList = node->AccessVector3(ABSORBTION);
        std::vector<Vector3> nodeScatList = node->AccessVector3(SCATTERING);
        std::vector<float> nodeIORList = node->AccessFloat(IOR);
        std::vector<float> nodePhaseList = node->AccessFloat(PHASE);

        absorbtionList.insert(absorbtionList.end(), nodeAbsList.begin(), nodeAbsList.end());
        scaterringList.insert(scaterringList.end(), nodeScatList.begin(), nodeScatList.end());
        iorList.insert(iorList.end(), nodeIORList.begin(), nodeIORList.end());
        phaseList.insert(phaseList.end(), nodePhaseList.begin(), nodePhaseList.end());

        // Id list on load
        const auto& nodeIdList = node->Ids();
        for(const auto& id : nodeIdList)
        {
            idList.push_back(id.second);
        }

    }


    // Finally 
   
	return SceneError::OK;
}

SceneError CPUMediumHomogenous::ChangeTime(const NodeListing& transformNodes, double time,
										   const std::string& scenePath)
{
	return SceneError::OK;
}

TracerError CPUMediumHomogenous::ConstructMediums(const CudaSystem&)
{
	return TracerError::OK;
}