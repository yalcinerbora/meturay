
cmake -G "Visual Studio 17 2022" -A x64 -S .\Build\CMake\ -B Bin\CMake -DUSE_OPTIX=ON
cmake --build Bin\CMake --config Release --parallel
cd WorkingDir
# Run Normal Version
echo "Running Non-product Version"
..\Bin\Win\Release\MRay.exe -t .\tConf_veach_normal.json -v .\vConf_veach_normal.json -r 1920x1080 .\Scenes\veachDoor-sun.json
# Run Product Version
echo "Running Product Version"
..\Bin\Win\Release\MRay.exe -t .\tConf_veach_product.json -v .\vConf_veach_product.json -r 1920x1080 .\Scenes\veachDoor-sun.json
# Run the FLIP to generate mean flip values
python ..\Python\flip.py -r .\veachDoor-sun-ref.exr -t .\veachDoor-wfpg-non-product\out_time_60.00s.exr .\veachDoor-wfpg-product\out_time_60.00s.exr
cd ..