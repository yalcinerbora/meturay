#include "SharedLib.h"
#include "System.h"
#include "Log.h"
#include "RayLib/FileSystemUtility.h"


// Env Headers
#if defined METURAY_WIN
    #include <windows.h>
    #include <strsafe.h>
#elif defined METURAY_LINUX
    #include <dlfcn.h>
#endif

#if defined METURAY_WIN
    std::wstring ConvertWinWchar(const std::string& unicodeStr)
    {

            const size_t length = unicodeStr.length();
            const DWORD kFlags = MB_ERR_INVALID_CHARS;

            // Quarry string size
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
    }
#endif

void* SharedLib::GetProcAdressInternal(const std::string& fName)
{
    #if defined METURAY_WIN
        return (void*)GetProcAddress((HINSTANCE)libHandle, fName.c_str());
    #elif defined METURAY_LINUX
        void* result = dlsym(libHandle, fName.c_str());
        if(result == nullptr)
            METU_ERROR_LOG("{}", dlerror());
        return result;
    #endif
}

SharedLib::SharedLib(const std::string& libName)
{
    std::string libWithExt = libName;
    #if defined METURAY_WIN
        libWithExt += WinDLLExt;
        libHandle = (void*)LoadLibrary(ConvertWinWchar(libWithExt).c_str());
        if(libHandle == nullptr)
        {
            DWORD errorMessageID = GetLastError();

            LPSTR messageBuffer = nullptr;
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                         FORMAT_MESSAGE_FROM_SYSTEM |
                                         FORMAT_MESSAGE_IGNORE_INSERTS,
                                         NULL, errorMessageID,
                                         MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                         (LPSTR)&messageBuffer, 0, NULL);

            // Get the buffer
            std::string message(messageBuffer, size);
            METU_ERROR_LOG(message);
            LocalFree(messageBuffer);
        }
    #elif defined METURAY_LINUX
        libWithExt = "lib";
        libWithExt += libName;
        libWithExt += LinuxDLLExt;

        // On Linux Directly provide the path
        // TODO: Change this to a more generic solution
        std::string execPath = Utility::CurrentExecPath();
        libWithExt = Utility::MergeFileFolder(execPath, libWithExt);
        libHandle = dlopen(libWithExt.c_str(), RTLD_NOW);
        if(libHandle == nullptr)
            METU_ERROR_LOG("{}", dlerror());
    #endif

    if(libHandle == nullptr)
        throw DLLException(DLLError::DLL_NOT_FOUND);
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