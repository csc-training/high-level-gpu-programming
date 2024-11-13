# Enumerate devices
When running a multi-gpu program it is very important that each process or  cpu thread picks the right devices. SYCL provides several function to check the visible devices. For example the `get_devices` which is part of the `sycl::devices` namespace can obtain the list of devices which follow a specific criteria. With the argument `sycl::info::device_type::gpu` it will list all device which have the `device_type` property `gpu`. Similarly one can get the available CPU using the keyword `cpu`as argument. 
A complete code is shown in the  [enumerate_device.cpp](enumerate_device.cpp).

Alternatively, one can loop through all platforms,checking all devices available on each platform:
```
for (const auto & p : platform::get_platforms()) {
    for (const auto& d: p.get_devices()) {
        std::cout << "name: " << d.get_info<info::device::name>() << std::endl;
    }  
}
``` 

The `sycl::info` namespace is quite wide. It is possible to query platform name and version, device name, and also some low level details such as maximum size permitted for work-groups or how many compute units (or SMP) the device has.
