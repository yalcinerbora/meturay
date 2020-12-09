#pragma once

namespace TracerFunctions
{
    __device__
    inline float FrenelDielectric(float cosIn, float iorIn, float iorOut)
    {
        // Calculate Sin from Snell's Law
        float sinIn = sqrt(max(0.0f, 1.0f - cosIn * cosIn));
        float sinOut = iorIn / iorOut * sinIn;

        // Total internal reflection
        if(sinOut >= 1.0f) return 1.0f;

        // Frenel Equation
        float cosOut = sqrt(max(0.0f, 1.0f - sinOut * sinOut));

        float parallel = ((iorOut * cosIn - iorIn * cosOut) /
                          (iorOut * cosIn + iorIn * cosOut));
        parallel = parallel * parallel;

        float perpendicular = ((iorIn * cosIn - iorOut * cosOut) /
                               (iorIn * cosIn + iorOut * cosOut));
        perpendicular = perpendicular * perpendicular;
        
        return (parallel + perpendicular) * 0.5f;
    }
}