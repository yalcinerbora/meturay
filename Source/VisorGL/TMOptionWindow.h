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

        const ToneMapOptions&   TMOptions() const;

        void                    Render() override;
        bool                    IsWindowClosed() const override;
};

inline TMOptionWindow::TMOptionWindow()
    : opts{DefaultTMOptions}
    , windowClosed(true)
{}

inline TMOptionWindow::TMOptionWindow(bool windowClosed,
                                      const ToneMapOptions& defaultOptions)
    : opts(defaultOptions)
    , windowClosed(windowClosed)
{}

inline const ToneMapOptions& TMOptionWindow::TMOptions() const
{
    return opts;
}

inline bool TMOptionWindow::IsWindowClosed() const
{
    return windowClosed;
}