#pragma once
/**

Many Constants that are used throught the tracer

*/

namespace SceneConstants
{
	// Fundamental Limitations (for convenience)
	static constexpr const int	MaxSurfacePerAccelerator = 8;
}

namespace MathConstants
{
	static constexpr double	Pi_d = 3.1415926535897932384626433;
	static constexpr double PiSqr_d = Pi_d * Pi_d;
	static constexpr double InvPi_d = 1.0 / Pi_d;
	static constexpr double InvPiSqr_d = 1.0 / (Pi_d * Pi_d);
	static constexpr double Sqrt2_d = 1.4142135623730950488016887;
	static constexpr double Sqrt3_d = 1.7320508075688772935274463;
	static constexpr double E_d = 2.7182818284590452353602874;
	static constexpr double InvE_d = 1.0 / E_d;

	static constexpr double DegToRadCoef_d = Pi_d / 180.0;
	static constexpr double RadToDegCoef_d = 180.0 / Pi_d;

	static constexpr double Epsilon_d = 0.00001;

	static constexpr float Pi = static_cast<float>(Pi_d);
	static constexpr float PiSqr = static_cast<float>(PiSqr_d);
	static constexpr float InvPi = static_cast<float>(InvPi_d);
	static constexpr float InvPiSqr = static_cast<float>(InvPiSqr_d);
	static constexpr float Sqrt2 = static_cast<float>(Sqrt2_d);
	static constexpr float Sqrt3 = static_cast<float>(Sqrt3_d);
	static constexpr float E = static_cast<float>(E_d);
	static constexpr float InvE = static_cast<float>(InvE_d);

	static constexpr float DegToRadCoef = static_cast<float>(DegToRadCoef_d);
	static constexpr float RadToDegCoef = static_cast<float>(RadToDegCoef_d);

	static constexpr float Epsilon = static_cast<float>(Epsilon_d);
}
