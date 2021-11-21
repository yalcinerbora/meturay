#pragma once

#include <bitset>

template<class Enum, int S> class Flags;
template <class Enum, int S> Flags<Enum, S> operator|(Enum e1, Enum e2);

template<class Enum, int S = static_cast<int>(Enum::END)>
class Flags
{
    public:
        // I mean... lets just be sure
        static_assert(sizeof(Enum) <= sizeof(size_t),
                      "Flags -> Enum may have more than size_t amount of data");

        using F = Enum;


    private:
        std::bitset<S> flagData;

    protected:

    public:
        // Constructors & Destructor
                        Flags() = default;
                        Flags(Enum);
                        template <int C>
                        Flags(const std::array<Enum, C>& vals);
                        Flags(const Flags&) = default;
                        Flags(Flags&&) = default;
        Flags&          operator=(const Flags&) = default;
        Flags&          operator=(Flags&&) = default;
                        ~Flags() = default;

        bool&           operator[](Enum);
        bool            operator[](Enum) const;

        Flags&          operator|(Enum);
        Flags&          operator|=(Enum);

        template<class E, int Sz>
        friend Flags    operator|(E, E);
};

template <class Enum, int S = static_cast<int>(Enum::END)>
Flags<Enum, S> operator|(Enum e1, Enum e2)
{
    return Flags<Enum, S>(std::array<Enum, 2>{e1, e2});
}

template<class Enum, int S>
Flags<Enum, S>::Flags(Enum e)
{
    flagData[static_cast<size_t>(e)] = true;
}

template<class Enum, int S>
template <int C>
Flags<Enum, S>::Flags(const std::array<Enum, C>& vals)
{
    for(Enum e : vals)
        flagData.set(static_cast<size_t>(e));
}

template<class Enum, int S>
bool& Flags<Enum, S>::operator[](Enum e)
{
    return flagData[static_cast<size_t>(e)];
}

template<class Enum, int S>
bool Flags<Enum, S>::operator[](Enum e) const
{
    return flagData[static_cast<size_t>(e)];
}

template<class Enum, int S>
Flags<Enum, S>& Flags<Enum, S>::operator|(Enum e)
{
    // Set is fine here
    flagData.set(static_cast<size_t>(e));
    return *this;
}

template<class Enum, int S>
Flags<Enum, S>& Flags<Enum, S>::operator|=(Enum e)
{
    // Set is fine here
    flagData.set(static_cast<size_t>(e));
    return *this;
}