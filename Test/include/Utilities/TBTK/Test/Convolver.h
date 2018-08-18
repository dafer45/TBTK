#include "TBTK/Convolver.h"

#include "gtest/gtest.h"

namespace TBTK{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

TEST(Convolver, convolve){
	//Setup the input.
	Array<std::complex<double>> array0({10, 5});
	Array<std::complex<double>> array1({10, 5});
	for(unsigned int x = 0; x < 10; x++){
		for(unsigned int y = 0; y < 5; y++){
			array0[{x, y}] = (double)x*y + std::complex<double>(0, 1);
			array1[{x, y}] = x+y;
		}
	}

	//Perform the convolution.
	Array<std::complex<double>> result
		= Convolver::convolve(array0, array1);

	//Calculate reference result.
	Array<std::complex<double>> reference({10, 5}, 0);
	for(unsigned int x = 0; x < 10; x++){
		for(unsigned int y = 0; y < 5; y++){
			for(unsigned int xp = 0; xp < 10; xp++){
				for(unsigned int yp = 0; yp < 5; yp++){
					reference[{x, y}]
						+= array0[
							{xp, yp}
						]*array1[
							{
								((10+x) - xp)%10,
								((5+y) - yp)%5
							}
						];
				}
			}
		}
	}

	//Check the result against the reference.
	for(unsigned int x = 0; x < 10; x++){
		for(unsigned int y = 0; y < 5; y++){
			EXPECT_NEAR(
				real(result[{x, y}]),
				real(reference[{x, y}]),
				EPSILON_100
			);
			EXPECT_NEAR(
				imag(result[{x, y}]),
				imag(reference[{x, y}]),
				EPSILON_100
			);
		}
	}
}

TEST(Convolver, crossCorrelate){
	//Setup the input.
	Array<std::complex<double>> array0({10, 5});
	Array<std::complex<double>> array1({10, 5});
	for(unsigned int x = 0; x < 10; x++){
		for(unsigned int y = 0; y < 5; y++){
			array0[{x, y}] = (double)x*y + std::complex<double>(0, 1);
			array1[{x, y}] = x+y;
		}
	}

	//Perform the convolution.
	Array<std::complex<double>> result
		= Convolver::crossCorrelate(array0, array1);

	//Calculate reference result.
	Array<std::complex<double>> reference({10, 5}, 0);
	for(unsigned int x = 0; x < 10; x++){
		for(unsigned int y = 0; y < 5; y++){
			for(unsigned int xp = 0; xp < 10; xp++){
				for(unsigned int yp = 0; yp < 5; yp++){
					reference[{x, y}]
						+= conj(
							array0[{xp, yp}]
						)*array1[
							{
								(xp + x)%10,
								(yp + y)%5
							}
						];
				}
			}
		}
	}

	//Check the result against the reference.
	for(unsigned int x = 0; x < 10; x++){
		for(unsigned int y = 0; y < 5; y++){
			EXPECT_NEAR(
				real(result[{x, y}]),
				real(reference[{x, y}]),
				EPSILON_100
			);
			EXPECT_NEAR(
				imag(result[{x, y}]),
				imag(reference[{x, y}]),
				EPSILON_100
			);
		}
	}
}

};
