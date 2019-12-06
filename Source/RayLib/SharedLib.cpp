#include "SharedLib.h"
#include "System.h"

// Env Headers
#if defined METURAY_WIN
    #include <windows.h>
    #include <strsafe.h>
#elif defined METURAY_LINUX
    #include <dlfcn.h>
#endif

std::wstring ConvertWinWchar(const std::string& unicodeStr)
{
    #if defined METURAY_WIN
        const size_t length = unicodeStr.length();
        const DWORD kFlags = MB_ERR_INVALID_CHARS;

        // Quaery string size
        const int utf16Length = ::MultiByteToWideChar(
            CP_UTF8,                    // Source string is in UTF-8
            kFlags,                     // Conversion flags
            unicodeStr.data(),          // Source UTF-8 string pointer
            static_cast<int>(length),   // Length of the source UTF-8 string, in chars
            nullptr,                    // Unused - no conversion done in this step
            0                           // Request size of destination buffer, in wchar_ts
        );

        std::wstring wString(utf16Length, L'\0');

        // Convert from UTF-8 to UTF-16
        ::MultiByteToWideChar(
            CP_UTF8,                    // Source string is in UTF-8
            kFlags,                     // Conversion flags
            unicodeStr.data(),          // Source UTF-8 string pointer
            static_cast<int>(length),   // Length of source UTF-8 string, in chars
            wString.data(),             // Pointer to destination buffer
            utf16Length                 // Size of destination buffer, in wchar_ts
        );
        return wString;
    #elif defined METURAY_LINUX
        return std::wstring();
    #endif
}


void* SharedLib::GetProcAdressInternal(const std::string& fName)
{
    #if defined METURAY_WIN
        return (void*)GetProcAddress((HINSTANCE)libHandle, fName.c_str());
    #elif defined METURAY_LINUX
        return dlsym(Lib, Fnname);
    #endif
}

SharedLib::SharedLib(const std::string& libName)
{
    std::string libWithExt = libName;
    #if defined METURAY_WIN
        libWithExt += WinDLLExt;
        libHandle = (void*)LoadLibrary(ConvertWinWchar(libWithExt).c_str());
    #elif defined METURAY_LINUX
        libWithExt += LinuxDLLExt;
        libHandle = dlopen(libWithExt.c_str(), iMode, RTLD_NOW);        
    #endif

    // Check if we found the DLL
    if(libHandle == nullptr)
    {
        LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;
        DWORD dw = GetLastError();

        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            dw,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0, NULL);

        // Display the error message and exit the process

        lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
            (lstrlen((LPCTSTR)lpMsgBuf) + 100) * sizeof(TCHAR));
        StringCchPrintf((LPTSTR)lpDisplayBuf,
                        LocalSize(lpDisplayBuf) / sizeof(TCHAR),
                        TEXT("Failed with error %d: %s"),
                        dw, lpMsgBuf);
        MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);
        throw DLLException(DLLError::DLL_NOT_FOUND);
    }
}

SharedLib::~SharedLib()
{
    #if defined METURAY_WIN
        if(libHandle != nullptr) FreeLibrary((HINSTANCE)libHandle);
    #elif defined METURAY_LINUX
        if(libHandle != nullptr) dlclose(libHandle);
    #endif
}

SharedLib::SharedLib(SharedLib&& other) noexcept
    : libHandle(other.libHandle)
{
    other.libHandle = nullptr;
}

SharedLib& SharedLib::operator=(SharedLib&& other) noexcept
{
    libHandle = other.libHandle;
    other.libHandle = nullptr;
    return *this;
}