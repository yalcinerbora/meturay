#pragma once
/**

From:
https://akrzemi1.wordpress.com/2017/06/28/compile-time-string-concatenation/
https://stackoverflow.com/questions/60593058/is-it-possible-to-concatenate-two-strings-of-type-const-char-at-compile-time

*/

#include <cassert>
#include <type_traits>

template<std::size_t N>
class StaticString
{
    private:
        char state[N+1] = {0};

    public:
        // Constructors & Destructor
        constexpr StaticString(const char(&arr)[N+1] )
        {
            for (std::size_t i = 0; i < N; ++i)
                state[i] = arr[i];
        }
        constexpr StaticString() = default;
        constexpr StaticString(const StaticString&) = default;
        constexpr StaticString& operator=(const StaticString&) = default;

        constexpr char          operator[](std::size_t i) const { return state[i]; }
        constexpr char&         operator[](std::size_t i) { return state[i]; }

        constexpr explicit      operator const char*() const { return state; }
        constexpr std::size_t   Size() const { return N; }
        constexpr char const*   Begin() const { return state; }
        constexpr char const*   End() const { return Begin() + Size(); }

        template<std::size_t M>
        friend constexpr StaticString<N+M> operator+(StaticString lhs,
                                                     StaticString<M> rhs )
        {
            StaticString<N+M> retval;
            for (std::size_t i = 0; i < N; ++i)
                retval[i] = lhs[i];
            for (std::size_t i = 0; i < M; ++i)
                retval[N+i] = rhs[i];
            return retval;
        }

        friend constexpr bool operator==(StaticString lhs, StaticString rhs )
        {
            for (std::size_t i = 0; i < N; ++i)
                if (lhs[i] != rhs[i]) return false;
            return true;
        }
        friend constexpr bool operator!=(StaticString lhs, StaticString rhs )
        {
            for (std::size_t i = 0; i < N; ++i)
                if (lhs[i] != rhs[i]) return true;
            return false;
        }
        template<std::size_t M, std::enable_if_t< M!=N, bool > = true>
        friend constexpr bool       operator!=(StaticString lhs, StaticString<M> rhs ) { return true; }
        template<std::size_t M, std::enable_if_t< M!=N, bool > = true>
        friend constexpr bool       operator==(StaticString, StaticString<M> ) { return false; }
};

template<std::size_t N>
StaticString( char const(&)[N] )->StaticString<N-1>;