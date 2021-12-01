#pragma once

#include "VisorGUIWindowI.h"
#include "Structs.h"

class AnalyticData;
class SceneAnalyticData;
class TracerState;

class InfoPanel : public VisorGUIWindowI
{
    private:


    protected:
    public:
                    InfoPanel(const ToneMapOptions&,
                              const SceneAnalyticData&,
                              const TracerState&);
                    ~InfoPanel() = default;

    void            Render() override;

};