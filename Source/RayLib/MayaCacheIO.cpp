#include "MayaCacheIO.h"
#include "Endian.h"
#include "Log.h"

#include <rapidxml.hpp>
#include <fstream>
#include <regex>
#include <map>

#include <filesystem>

namespace MayaCache
{
    // Inner Tags
    static constexpr const char* FOR4 = "FOR4";
    static constexpr const char* FOR8 = "FOR8";
    //static constexpr const char* MYCH = "MYCH";

    struct NSChannelHeader
    {
        enum NSChannelLogic
        {
            NS_DENSITY,
            NS_VELOCITY,
            NS_UNNECESARY
        };
        enum DataChannelType
        {
            FVCA,       // Float Vector Array (3 component)
            DVCA,       // Double Vector Array (3 component)
            FBCA,       // Float Array
            DBLA,       // Double Array
            DATA_CHANNEL_END
        };
        static constexpr const char* DataChannelNames[] =
        {
            "FVCA",
            "DVCA",
            "FBCA",
            "DBLA"
        };
        static constexpr int DataTypeSizes[] =
        {
            sizeof(float) * 3,
            sizeof(double) * 3,
            sizeof(float),
            sizeof(double)
        };
        static_assert(sizeof(DataChannelNames) / sizeof(const char*) == DATA_CHANNEL_END, "Data Channel Type Mismatch.");
        static_assert(sizeof(DataTypeSizes) / sizeof(int) == DATA_CHANNEL_END, "Data Channel Type Mismatch.");

        // Members
        NSChannelLogic      logic;
        DataChannelType     type;
        size_t              byteSize;
        size_t              fileDataLoc;
    };

    std::string ParseNCacheFirstName(const std::string& s)
    {
        if(s.find_first_of('.') == s.find_last_of('.'))
        {
            return s.substr(s.find_first_of('.') + 1,
                            s.find_first_of('=') - s.find_first_of('.') - 1);
        }
        else
        {
            std::string sNext = s.substr(s.find_first_of('.') + 1);
            return sNext.substr(0, sNext.find_first_of('.'));
        }
    }

    std::string ParseNCacheSecondName(const std::string& s)
    {
        std::string sNext = s.substr(s.find_first_of('.') + 1);
        sNext = sNext.substr(sNext.find_first_of('.') + 1);
        return sNext.substr(0, sNext.find_first_of('='));
    }

    template<int WordSize>
    std::string ReadTag(std::ifstream& f)
    {
        std::string out(WordSize, '\0');
        f.read(out.data(), WordSize);

        if constexpr (WordSize == 8)
            out = out.substr(0, 4);
        return out;
    }

    template<class T, int WordSize = sizeof(T)>
    T ReadInt(std::ifstream& file)
    {
        T out;
        file.read(reinterpret_cast<char*>(&out), WordSize);

        if constexpr(WordSize == 8)
        {
            return be64toh(out);
        }
        else if constexpr(WordSize == 4)
        {
            return be32toh(out);
        }
        else return out;
    }

    float ReadFloat(std::ifstream& file)
    {
        static_assert(sizeof(int32_t) == sizeof(float));

        int32_t out;
        file.read(reinterpret_cast<char*>(&out), sizeof(float));
        out = be32toh(out);
        return *reinterpret_cast<float*>(&out);
    }

    template<class T, int WordSize = sizeof(T)>
    IOError LoadCacheNavierStokes(std::vector<float>& velocityDensityData,
                                  const MayaNSCacheInfo& info,
                                  std::ifstream& file)
    {
        static_assert(WordSize == 4 || WordSize == 8);
        // File stream should already read the first 4 byte
        if constexpr(WordSize == 8)
            file.seekg(4, std::ifstream::cur);

        // Read Header Size and Skip
        T blockSize = ReadInt<T>(file);
        file.seekg(blockSize, std::ifstream::cur);

        // Read another FOR4 or FOR8 tag (Maya Rocks!!!!)
        std::string tagFOURCC = ReadTag<WordSize>(file);
        if((tagFOURCC != FOR4 || WordSize != 4) &&
           (tagFOURCC != FOR8 || WordSize != 8))
        {
            return IOError::NCACHE_INVALID_FOURCC;
        }

        // Read Data Size
        //T dataSize = ReadInt<T>(file);
        // Read MYCH
        std::string tagMYCH = ReadTag<4>(file);

        // Read First Data Part
        std::vector<NSChannelHeader> channels;
        //size_t initalPtr = file.tellg();
        //for(size_t currentPtr = file.tellg();
        //  (currentPtr - initalPtr) < static_cast<size_t>(dataSize);
        //  currentPtr = file.tellg())
        while(file.peek() != EOF)
        {
            NSChannelHeader header;
            if(IOError e; (e = LoadCacheDataHeader<T>(header, file)) != IOError::OK)
            {
                return e;
            }

            // Save Header
            channels.push_back(header);
            // Goto next Header
            size_t paddedByteSize = (header.byteSize + WordSize - 1) / WordSize * WordSize;
            file.seekg(paddedByteSize, std::ifstream::cur);
        }

        // Set size
        const size_t totalCount = info.dim[0] * info.dim[1] * info.dim[2];
        velocityDensityData.resize(totalCount * 4);

        // Channel Headers read. Now read actual data
        // Unfortunately we need to read word by word
        for(const NSChannelHeader& h : channels)
        {
            size_t offset = 0;
            size_t stride = 4;
            if(h.logic == NSChannelHeader::NS_DENSITY)
            {
                offset = 3;
            }
            else if(h.logic == NSChannelHeader::NS_VELOCITY)
            {
            }
            else continue;

            // Seek fileptr there
            file.seekg(h.fileDataLoc);

            for(size_t i = 0; i < totalCount; i++)
            {
                if(h.logic == NSChannelHeader::NS_DENSITY)
                {
                    velocityDensityData[offset + i * stride] = ReadFloat(file);
                }
                else
                {
                    velocityDensityData[i * stride    ] = ReadFloat(file);
                    velocityDensityData[i * stride + 1] = ReadFloat(file);
                    velocityDensityData[i * stride + 2] = ReadFloat(file);
                }
            }
        }
        return IOError::OK;
    }

    template<class T, int WordSize = sizeof(T)>
    IOError LoadCacheDataHeader(NSChannelHeader& header,
                                std::ifstream& file)
    {
        std::string tagCHNM = ReadTag<WordSize>(file);
        // Channel Name (Padded with 32/64 bit boundaries)
        T channelNameSize = ReadInt<T>(file);
        T channelNameSizePadded = (channelNameSize + WordSize - 1) / WordSize * WordSize;
        std::string channelName(channelNameSizePadded, '\0');
        file.read(channelName.data(), channelNameSize);
        file.seekg(channelNameSizePadded - channelNameSize, std::ifstream::cur);
        // Determine Channel Logic by Name
        std::string channelLogicString = channelName.substr(channelName.find_last_of('_') + 1,
                                                            channelName.find_first_of('\0') - channelName.find_last_of('_') - 1);
        if(channelLogicString == "density")
        {
            header.logic = NSChannelHeader::NS_DENSITY;
        }
        else if(channelLogicString == "velocity")
        {
            header.logic = NSChannelHeader::NS_VELOCITY;
        }
        else header.logic = NSChannelHeader::NS_UNNECESARY;

        // Size part
        std::string tagSIZE = ReadTag<WordSize>(file);
        ReadInt<T>(file);
        [[maybe_unused]] T channelDataCount = ReadInt<int32_t>(file);
        // Dummy Read (Entire Data format is super inconsistent...)
        if constexpr(WordSize == 8) ReadInt<uint32_t>(file);
        // Format Tag
        std::string tagFORMAT = ReadTag<WordSize>(file);
        // Byte Size
        T bufferByteSize = ReadInt<T>(file);
        header.byteSize = static_cast<size_t>(bufferByteSize);
        // Assert that Data Count and Byte Size is ok
        if(tagFORMAT == NSChannelHeader::DataChannelNames[NSChannelHeader::FVCA])
        {
            assert(static_cast<unsigned int>(bufferByteSize) == channelDataCount * 3u * sizeof(float));
            header.type = NSChannelHeader::FVCA;
        }
        else if(tagFORMAT == NSChannelHeader::DataChannelNames[NSChannelHeader::DVCA])
        {
            assert(static_cast<unsigned int>(bufferByteSize) == channelDataCount * 3u * sizeof(double));
            header.type = NSChannelHeader::DVCA;
        }
        else if(tagFORMAT == NSChannelHeader::DataChannelNames[NSChannelHeader::FBCA])
        {
            assert(static_cast<unsigned int>(bufferByteSize) == channelDataCount * sizeof(float));
            header.type = NSChannelHeader::FBCA;
        }
        else if(tagFORMAT == NSChannelHeader::DataChannelNames[NSChannelHeader::DBLA])
        {
            assert(static_cast<unsigned int>(bufferByteSize) == channelDataCount * sizeof(double));
            header.type = NSChannelHeader::DBLA;
        }
        else return IOError::NCACHE_INVALID_FORMAT;

        // Current ifstream ptr is on the data part now
        // Save it
        header.fileDataLoc = file.tellg();
        return IOError::OK;
    }
}

std::u8string MayaCache::GenerateNCacheFrameFile(const std::u8string& xmlFile, int frameNo)
{
    std::filesystem::path path = std::filesystem::path(xmlFile);
    std::filesystem::path fileNameOnly = path.stem();
    fileNameOnly += (std::string("Frame") + std::to_string(frameNo) + std::string(".mcx"));
    std::filesystem::path framePath = path.parent_path() / fileNameOnly;
    return framePath.u8string();
}

IOError MayaCache::LoadNCacheNavierStokesXML(MayaNSCacheInfo& info,
                                             const std::u8string& fileName)
{
    static constexpr const char* ExtraTag = "extra";

    size_t size = std::filesystem::file_size(fileName);
    if(size == std::numeric_limits<std::uintmax_t>::max())
        return IOError::FILE_NOT_FOUND;

    std::ifstream file = std::ifstream(std::filesystem::path(fileName));
    if(!file.is_open()) return IOError::FILE_NOT_FOUND;

    // Parse XML
    std::vector<char> xmlFile(size + 1);
    file.read(xmlFile.data(), xmlFile.size());
    xmlFile.back() = '\0';

    rapidxml::xml_document<> xml;
    xml.parse<rapidxml::parse_default>(xmlFile.data());

    // Sorted Interpolations
    std::map<float, float> opacities;
    std::map<float, Vector3f> colors;

    // Parse "extra" tags
    const auto mainNode = xml.first_node();
    for(auto node = mainNode->first_node(ExtraTag);
        node != nullptr;
        node = node->next_sibling("extra"))
    {
        // Strip Node Name and value
        std::string nodeName = ParseNCacheFirstName(node->value());
        std::string nodeValue = node->value();
        nodeValue = nodeValue.substr(nodeValue.find_last_of('=') + 1);

        // Check
        if(nodeName == "resolutionW")
        {
            info.dim[0] = std::stoi(nodeValue);
        }
        else if(nodeName == "resolutionH")
        {
            info.dim[1] = std::stoi(nodeValue);
        }
        else if(nodeName == "resolutionD")
        {
            info.dim[2] = std::stoi(nodeValue);
        }
        // Length
        else if(nodeName == "dimensionsW")
        {
            info.size[0] = std::stof(nodeValue);
        }
        else if(nodeName == "dimensionsH")
        {
            info.size[1] = std::stof(nodeValue);
        }
        else if(nodeName == "dimensionsD")
        {
            info.size[2] = std::stof(nodeValue);
        }
        // Transparency
        else if(nodeName == "transparencyR")
        {
            info.transparency[0] = std::stof(nodeValue);
        }
        else if(nodeName == "transparencyG")
        {
            info.transparency[1] = std::stof(nodeValue);
        }
        else if(nodeName == "transparencyB")
        {
            info.transparency[2] = std::stof(nodeValue);
        }
        // Regex Parsing
        std::string subscript = "\\[[0-9*]\\]";
        if(std::regex_match(nodeName, std::regex("color" + subscript)))
        {
            float interp;
            Vector3f color;

            // Iterate all color values
            int index = std::stoi(nodeName.substr(nodeName.find_first_of("[") + 1,
                                                  nodeName.find_first_of("]") - nodeName.find_first_of("[") - 1));
            while(std::regex_match(nodeName, std::regex("color\\[" + std::to_string(index) + "\\]")))
            {
                std::string nodeSecondName = ParseNCacheSecondName(node->value());
                if(nodeSecondName == "color_Position")
                {
                    interp = std::stof(nodeValue);
                }
                else if(nodeSecondName == "color_ColorR")
                {
                    color[0] = std::stof(nodeValue);
                }
                else if(nodeSecondName == "color_ColorG")
                {
                    color[1] = std::stof(nodeValue);
                }
                else if(nodeSecondName == "color_ColorB")
                {
                    color[2] = std::stof(nodeValue);
                }
                // Advance
                node = node->next_sibling("extra");
                nodeName = ParseNCacheFirstName(node->value());
                nodeValue = node->value();
                nodeValue = nodeValue.substr(nodeValue.find_last_of('=') + 1);
            }
            node = node->previous_sibling("extra");
            colors.emplace(interp, color);
        }
        else if(std::regex_match(nodeName, std::regex("opacity" + subscript)))
        {
            float interp, opacity;
            // Iterate all color values
            int index = std::stoi(nodeName.substr(nodeName.find_first_of("[") + 1,
                                                  nodeName.find_first_of("]") - nodeName.find_first_of("[") - 1));
            while(std::regex_match(nodeName, std::regex("opacity\\[" + std::to_string(index) + "\\]")))
            {
                std::string nodeSecondName = ParseNCacheSecondName(node->value());
                if(nodeSecondName == "opacity_Position")
                {
                    interp = std::stof(nodeValue);
                }
                else if(nodeSecondName == "opacity_FloatValue")
                {
                    opacity = std::stof(nodeValue);
                }
                // Advance
                node = node->next_sibling("extra");
                nodeName = ParseNCacheFirstName(node->value());
                nodeValue = node->value();
                nodeValue = nodeValue.substr(nodeValue.find_last_of('=') + 1);
            }
            node = node->previous_sibling("extra");
            opacities.emplace(interp, opacity);
        }
    }

    // Write to Arrays
    for(const auto& pair : colors)
    {
        info.color.push_back(pair.second);
        info.colorInterp.push_back(pair.first);
    }
    for(const auto& pair : opacities)
    {
        info.opacity.push_back(pair.second);
        info.opacityInterp.push_back(pair.first);
    }

    // Channels
    auto channelNode = mainNode->first_node("Channels");
    for(auto channel = channelNode->first_node();
        channel != nullptr;
        channel = channel->next_sibling())
    {
        std::string attrib = channel->first_attribute("ChannelName")->value();
        attrib = attrib.substr(attrib.find_first_of('_') + 1);

        if(attrib == "density")
        {
            info.channels.push_back(DENSITY);
        }
        else if(attrib == "velocity")
        {
            info.channels.push_back(VELOCITY);
        }
        else if(attrib == "resolution")
        {
            info.channels.push_back(RESOLUTION);
        }
        else if(attrib == "offset")
        {
            info.channels.push_back(OFFSET);
        }
    }
    return IOError::OK;
}

IOError MayaCache::LoadNCacheNavierStokes(std::vector<float>& velocityDensityData,
                                          const MayaNSCacheInfo& info,
                                          const std::u8string& fileName)
{
    static constexpr int FourCCSize = 4;
    std::ifstream file(std::filesystem::path(fileName),
                       std::iostream::binary);

    if(!file.is_open())
        return IOError::FILE_NOT_FOUND;

    // Read FOURCC
    //FOR4 or FOR8
    std::string fourCC(FourCCSize, '\0');
    file.read(fourCC.data(), FourCCSize);

    // Determine Read
    if(fourCC == FOR8)
    {
        return LoadCacheNavierStokes<int64_t>(velocityDensityData, info, file);
    }
    else if(fourCC == FOR4)
    {
        return LoadCacheNavierStokes<int32_t>(velocityDensityData, info, file);
    }
    else return IOError::NCACHE_INVALID_FOURCC;
}