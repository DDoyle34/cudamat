#include "../src/cudamat.h"
#include <stdio.h>

int main()
{
    // Creates a 3x3 matrix full of random floats
    Matrix a = init_rand(3, 3);

    // Creats a 3x3 matrix full of constant 0.6767
    Matrix b = init_const(3, 3, 0.6767);

    // Displays a and b
    mprint(a);
    mprint(b);

    // Adds a and b
    Matrix c = madd(a, b);
    mprint(c);

    // Multiplies and displayes a and b
    mprint(mmult(a, b));

    // Multiplies b by constant 0.8854 and displays
    mprint(kmult(0.8854, b));

    // Print a 5x5 identity matrix
    mprint(ident(5));

    // Performs the sigmoid function on all elements of a
    mprint(msigmoid(a));

    // Transposes d
    Matrix d = init_rand(3, 4);
    mprint(d);
    mprint(mtrans(d));

    mfree(b);
    mfree(c);
    unhandle();
}
