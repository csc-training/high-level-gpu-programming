# Enumerate devices
When running a multi-gpu program it is very important that each processor or  cpu thread picks the write device. SYCL provides several function to check the visible devices. For example the `get_devices` which is part of the `sycl::devices` namespace can obtain the list of devices which follow a specific criteria. With the argument `sycl::info::device_type::gpu` it will list all device which have the `device_type` property `gpu`. 
A complete code is shown in the  (enumerate_device.cpp).

