#pragma once

#include "CommandCallbacksI.h"

class VisorCallbacksI : public CommandCallbacksI
{
    public:
        virtual         ~VisorCallbacksI() = default;

        virtual void    WindowMinimizeAction(bool minimized) = 0;
        virtual void    WindowCloseAction() = 0;
};


class EmptyVisorCallback : public VisorCallbacksI
{
    void    ChangeScene(const std::u8string) override {}
    void    ChangeTime(const double) override {}
    void    IncreaseTime(const double) override {}
    void    DecreaseTime(const double) override {}

    void    ChangeCamera(const VisorTransform) override {}
    void    ChangeCamera(const unsigned int) override {}
    void    StartStopTrace(const bool) override {}
    void    PauseContTrace(const bool) override {}

    void    WindowMinimizeAction(bool) override {}
    void    WindowCloseAction() override {}
};