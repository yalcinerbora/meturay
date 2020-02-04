#pragma once
/**

From:
https://akrzemi1.wordpress.com/2017/06/28/compile-time-string-concatenation/

This is simpler version (without literal reference)
however it copies literals which means extra redundant storage

*/

#include <cassert>
#include <type_traits>

template <int N>
class static_string
{
    private:
        char    str[N + 1];

    protected:
    public:
        constexpr static_string(const char (&strIn)[N + 1])
        {
            assert(strIn[N] == '\0');
            for(int i = 0; i < N; ++i)
                str[i] = strIn[i];
            str[N] = '\0';
        }

        template <int M>
        constexpr static_string(const char(&s1)[M],
                                const char(&s2)[N - M])
         
        {
            for(int i = 0; i < M; ++i)
                str[i] = s1[i];

            for(int i = 0; i < N - M; ++i)
                str[M + i] = s2[i];

            str[N] = '\0';
        }

        constexpr char operator[](int i) const
        {
            assert(i >= 0 && i < N);
            return  str[i];
        }

        constexpr std::size_t size() const
        {
            return N;
        }

        constexpr const char* Str() const
        {
            return str;
        }
};

template <int N1, int N2>
constexpr static_string<N1 + N2> operator+(const static_string<N1>& s1,
                                           const static_string<N2>& s2)
{
    return static_string<N1 + N2>(s1, s2);
}

// Deduction guides for literal
template <int N>
static_string(const char(&)[N]) -> static_string<N - 1>;

template <int N, int M>
static_string(const char(&)[M],
              const char(&)[N - M]) -> static_string<N>;