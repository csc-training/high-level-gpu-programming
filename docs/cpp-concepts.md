---
title:  C++ refresher 
subtitle: High-Level GPU Programming 
author:   CSC Training
date:     2024-02
lang:     en
---

# C++ refresher{.section}

# Anatomy of a SYCL code

<small>
```cpp
using namespace sycl;

constexpr access::mode sycl_read = access::mode::read; constexpr access::mode sycl_write = access::mode::write;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N> &A, const std::array<T, N> &B, std::array<T, N> &C) {
  queue deviceQueue{property::queue::enable_profiling{}};
  range<1> numOfItems{N};
  buffer<T, 1> bufferA(A.data(), numOfItems); buffer<T, 1> bufferB(B.data(), numOfItems); buffer<T, 1> bufferC(C.data(), numOfItems);

  deviceQueue.submit([&](handler &cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh); auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class SimpleVadd<T>>(numOfItems,[=](id<1> wiID) {
      accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    });
  });
  deviceQueue.wait();
}
```
</small>

- SYCL and Kokkos are modern C++ with classes, templates, lambdas, ...

# Namespaces 

# Templates

# Raw pointers

- fundamental elements that store memory addresses of another variable. 
- **definition**: `float *ptrA = nullptr;`
- direct manipulation of memory.
- do not manage ownership or lifespan automatically.
- **allocation**: `ptrA= new float;`
- **deallocation**: `delete ptrA;`
- perform operations on pointers: `ptrA+N;` (shifts  by `N*sizeof(float)`).

# Structures

- composite data types which allows grouping of variables.

<div class="column">
<small>
```cpp
struct particle {
    double x;
    double y;
    double mass;
    int charge;
};
```
</small>

</div>

<div class="column">
<small>
```cpp
struct particle {
    double *x;
    double *y;
    double *mass;
    int *charge;
};
```
</small>
</div>
- access:  `double my_x=par_i.x;`. set: `par_i.x=5.0`. 
- structure can also  have functions (methods) as components:
<small>
```cpp
double compute_distance(const particle& par_j) const {
        double dx = x - par_j.x;
        double dy = y - par_j.y;
        return std::sqrt(dx * dx + dy * dy);
    }
...
particle par_i,par_j;
double distance=par_i.compute_distance(par_j);
```
</small>

# Classes

- similar to **structures**,  but members are either **public** or **private**

<div class="column">
<small>
```cpp
class particle {
private:
    double x;
    double y;
public:
    double *mass;
    int *charge;
    friend void set_function(particle &par, double newX, double newY); 
    void set_position(double newX, double newY) {
        x = newX;
        y = newY;
    }
    double get_x_position() { return x; }
};

```
</small>
</div>



<div class="column">
<small>
```cpp
void set_function(particle &par, double newX, double newY) {
    par.x = newX;
    par.y = newY;
}


...
particle par_i; 
par_i.set_position(5.0,6.0);  
// or
set_function(par_i,5.0,6.0); 
double my_x=par_i.get_x_position(); 
```
</small>
</div>


# Derived **classes/structures**

- **structures/classes** can have members other **structures/classes**.
- inherit properties and behaviors.
- within structures everything is public by default.
- withing classes  everything is private by default.

<div class="column">
<small>
```cpp
// Structure inheritance (public by default)
struct base_struct {...};

struct derived_struct : base_struct {...};
```
</small>
</div>


<div class="column">
<small>
```cpp
// Structure inheritance (public by default)
class base_class {...};

class derived_class : base_class {...};
```
</small>
</div>
- increases the code reusability and promotes a clean organization of the code.

# Lambdas

- anonymous function objects.
- a concise way to define small, unnamed functions inline 
- useful for short-lived tasks or as arguments to other functions.
- `[ capture clause ] (parameters) -> return-type { body }`
```cpp
int b = 6;
auto add = [=, &b](int a, int *b) -> int { return a + b[0]; };
int sum = add(5, &b);

```


# Functors

# Error Handling

# Summary