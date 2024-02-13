#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

// Buffers may not be directly accessed by the host and device (except
// through advanced and infrequently used mechanisms not described here).
// Instead, we must create accessors in order to read and write to buffers.
// Accessors provide the runtime with information about how we plan to use
// the data in buffers, allowing it to correctly schedule data movement.

// In this exercise the hostVector is declarated and initialized
// on the host.

// In the device block a buffer buf is created to interact with
// hostVector on the host.

//-----------------TASK-----------------------------
// Create buffer and an accessor to update the
// buffer buf on the device
// and write a h.parallel_for in which the indices
// are written to a and thereby also to buf
//-----------------TASK-----------------------------

int main()
{
    std::vector<int> hostVector(N, 3);

    std::cout << "printing hostVector before computation \n" ;
    for (int i = 0; i < N; i++) std::cout << hostVector[i] << " ";
    std::cout << "\n" ;

    {
        queue q;
        // Create a buffer buf for the hostVector
        // YOUR CODE GOES HERE
        buffer buf(hostVector);
        q.submit( [&] (handler& h)
         {
        // Create an accessor a for buf
        // and write a h.parallel_for in which the indices
        // are written to a and thereby also to buf
        // YOUR CODE GOES HERE
            accessor a(buf, h, write_only);
            h.parallel_for(N, [=] (id<1> i) { a[i] = i; } );
         } );

    }

    // When exiting the scope of the devices destroys the buffer
    // and updating the hostVector

    std::cout << "printing hostVector after computation \n" ;
    for (int i = 0; i < N; i++) std::cout << hostVector[i] << " ";
    std::cout << "\n" ;

    return 0;
}

