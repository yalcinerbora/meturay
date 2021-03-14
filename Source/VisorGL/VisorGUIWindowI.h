#pragma once

class VisorGUIWindowI
{
    public:
        virtual         ~VisorGUIWindowI() = default;

        // Interface
        virtual void    Render() = 0;
        virtual bool    IsWindowClosed() const = 0;
};