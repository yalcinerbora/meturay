#pragma once

#include "VisorGUIWindowI.h"
#include "Structs.h"

class TMOptionWindow : public VisorGUIWindowI
{
    private:
        ToneMapOptions          opts;
        bool                    windowClosed;

    protected:
    public:
        // Constructors & Destructor
                                TMOptionWindow();
                                TMOptionWindow(bool windowClosed,
                                               const ToneMapOptions& defaultOptions);
                                ~TMOptionWindow() = default;


        const ToneMapOptions&   GetToneMapOptions() const;

        void                    Render() override;
        bool                    IsWindowClosed() const override;
};

inline TMOptionWindow::TMOptionWindow()
    : windowClosed(true)
    , opts{}
{}

inline TMOptionWindow::TMOptionWindow(bool windowClosed,
                                      const ToneMapOptions& defaultOptions)
    : windowClosed(windowClosed)
    , opts(defaultOptions)
{}

inline const ToneMapOptions& TMOptionWindow::GetToneMapOptions() const
{
    return opts;
}

inline bool TMOptionWindow::IsWindowClosed() const
{
    return windowClosed;
}