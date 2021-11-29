#pragma once

class VisorGUIWindowI
{
    protected:
        bool            windowOpen = false;
    public:
        virtual         ~VisorGUIWindowI() = default;

        // Interface
        virtual void    Render() = 0;
        virtual bool    IsWindowOpen() const;
        virtual void    SetWindowOpen(bool);
        virtual void    ToggleWindowOpen();
};

inline bool VisorGUIWindowI::IsWindowOpen() const
{
    return windowOpen;
}
inline void VisorGUIWindowI::SetWindowOpen(bool b)
{
    windowOpen = b;
}

inline void VisorGUIWindowI::ToggleWindowOpen()
{
    windowOpen = !windowOpen;
}
