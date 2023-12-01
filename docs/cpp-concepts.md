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

- a way of organizing names (variables, functions, classes, etc.).
- access a name declared in a namespace: `my_namespace::my_variable`.
- the `using` directive` makes names available without the  `::` operator.
- for a specific name: `using my_namespace::my_function;`.
- for all names: `using namespace my_namespace;`.
- for `std`: `using namespace std;`.
- for `sycl`: `using namespace sycl;`.

# Templates

- allows writing generic programs
- entity that defines a family of classes, functions, variables, aliases, or concepts that can be parameterized 
```
template <class T>
T max (T a, T b) { // function template definition inside declaration
  return (a > b) ? a : b;
}
```
- `std::vector<float> A(N)`
- promotes code reusability and flexibility.
- support generic programming paradigms

# Raw pointers

- fundamental elements that store memory addresses of another variable. 
- **definition**: `float *ptrA = nullptr;`
- direct manipulation of memory.
- do not manage ownership or lifespan automatically.
- **allocation**: `ptrA= new float;`
- **deallocation**: `delete ptrA;`
- perform operations on pointers: `ptrA+N;` (shifts  by `N*sizeof(float)`).

# Classes

- composite data types which allows grouping of variables.
-  members are be **public** or **private**

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
par_i.mass=9.10938e-31;
```
</small>
</div>

- classes can  have *constructors* and *destructors*

# Derived classes

- **classes** can have members other **classes**.
- inherit properties and behaviors.
- withing classes  everything is private by default.


```cpp
// Class inheritance (private by default)
class base_class {...};

class derived_class : base_class {...};
```

- increases the code reusability 
- promotes a clean organization of the code.

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

- objects that behave as functions.
- can be encapsulated within classes.
- can be passed as an argument to another function.

<small>
```cpp
// Define the functor
class Add {
public:
    int operator()(int a, int b) {return a + b;}
};
void use_functor(Add add, int a, int b) {
    int sum = add(a, b);
    std::cout << "The sum is: " << sum << std::endl;
}

```
</small>

- can be used to achieve generic programming.
- allow us to write code that is more reusable, expressive, and efficient.

# Error Handling

- errors are handled via exceptions. 
- an unusual condition that results in an interruption in the flow of the program.
- C++ exceptions can cleanly separates the detection from  the handling.
- C++ exceptions can handle both *synchronous* and *asynchronous* errors.
sssss

<small>
```cpp
int main() {
  int x, y;
  cout << "Enter two numbers: ";
  cin >> x >> y;

  try {
    if (y == 0) {
      throw "Division by zero error"; // throw an exception
    }
    cout << "x / y = " << x / y << endl; // this may cause an exception
  }
  catch (const char* msg) {
    cerr << "Error: " << msg << endl; // catch and handle the exception
  }
  return 0;
}

```
</small>

# Summary

- SYCL and Kokkos are modern C++  aiming towards generic parallel programming. 
- classes, templates, lambdas, functors
- reusable, expressive, and efficient.
