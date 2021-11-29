#pragma once

#include "VisorGUIWindowI.h"
#include "Structs.h"

class TMOptionWindow : public VisorGUIWindowI
{
    private:
        ToneMapOptions          opts;

    protected:
    public:
        // Constructors & Destructor
                                TMOptionWindow();
                                TMOptionWindow(bool windowClosed,
                                               const ToneMapOptions& defaultOptions);
                                ~TMOptionWindow() = default;

        const ToneMapOptions&   TMOptions() const;

        void                    Render() override;
};

inline TMOptionWindow::TMOptionWindow()
    : opts{DefaultTMOptions}
{}

inline TMOptionWindow::TMOptionWindow(bool windowClosed,
                                      const ToneMapOptions& defaultOptions)
    : opts(defaultOptions)
{}

inline const ToneMapOptions& TMOptionWindow::TMOptions() const
{
    return opts;
}