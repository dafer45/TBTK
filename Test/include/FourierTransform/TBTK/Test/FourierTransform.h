#include "TBTK/Array.h"
#include "TBTK/FourierTransform.h"

#include "gtest/gtest.h"

namespace TBTK{

const double EPSILON_10000 = 10000*std::numeric_limits<double>::epsilon();
std::complex<double> i(0, 1);

//TODO:
//The Plan classes are not tested other than through the
//FourierTransform::transform() function. Complete testing therefore remains to
//be implemented. The number of transform functions should probably be reduces
//such that the functions with explicit dimension arguments sizeX, sizeY, sizeZ
//are removed in favor of the single method with a ranges argument. The number
//of tests for the Plan classes will be reduced if this is done first.

TEST(FourierTransform, transform1D){
	//Setup input array.
	Array<std::complex<double>> input({10});
	for(unsigned int x = 0; x < 10; x++)
		input[{x}] = x;

	//Create output array.
	Array<std::complex<double>> output({10});

	//Create reference array.
	Array<std::complex<double>> reference;

	for(int sign = -1; sign < 2; sign += 2){
		//Perform FFT.
		FourierTransform::transform(
			input.getData(),
			output.getData(),
			10,
			sign
		);

		//Calculate reference solution.
		reference = Array<std::complex<double>>({10}, 0);
		for(unsigned int k = 0; k < 10; k++){
			for(unsigned int x = 0; x < 10; x++){
				double K = 2*M_PI*(k/10.);
				reference[{k}] += input[{x}]*exp(
					((double)sign)*i*K*(double)x
				);
			}
		}
		for(unsigned int k = 0; k < 10; k++)
			reference[{k}] /= sqrt(10.);

		//Check the output against the reference solution.
		for(unsigned int k = 0; k < 10; k++){
			EXPECT_NEAR(
				real(output[{k}]),
				real(reference[{k}]),
				EPSILON_10000
			);
			EXPECT_NEAR(
				imag(output[{k}]),
				imag(reference[{k}]),
				EPSILON_10000
			);
		}
	}
}

TEST(FourierTransform, transform2D){
	//Setup input array.
	Array<std::complex<double>> input({10, 5});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			input[{x, y}] = x*y;

	//Create output array.
	Array<std::complex<double>> output({10, 5});

	//Create reference array.
	Array<std::complex<double>> reference;

	for(int sign = -1; sign < 2; sign += 2){
		//Perform FFT.
		FourierTransform::transform(
			input.getData(),
			output.getData(),
			10,
			5,
			sign
		);

		//Calculate reference solution.
		reference = Array<std::complex<double>>({10, 5}, 0);
		for(unsigned int kx = 0; kx < 10; kx++){
			for(unsigned int ky = 0; ky < 5; ky++){
				for(unsigned int x = 0; x < 10; x++){
					for(unsigned int y = 0; y < 5; y++){
						double KX = 2*M_PI*(kx/10.);
						double KY = 2*M_PI*(ky/5.);
						reference[{kx, ky}] += input[
							{x, y}
						]*exp(
							((double)sign)*i*(
								KX*(double)x
								+ KY*(double)y
							)
						);
					}
				}
			}
		}
		for(unsigned int kx = 0; kx < 10; kx++)
			for(unsigned int ky = 0; ky < 5; ky++)
				reference[{kx, ky}] /= sqrt(10.*5.);

		//Check the output against the reference solution.
		for(unsigned int kx = 0; kx < 10; kx++){
			for(unsigned int ky = 0; ky < 5; ky++){
				EXPECT_NEAR(
					real(output[{kx, ky}]),
					real(reference[{kx, ky}]),
					EPSILON_10000
				);
				EXPECT_NEAR(
					imag(output[{kx, ky}]),
					imag(reference[{kx, ky}]),
					EPSILON_10000
				);
			}
		}
	}
}

TEST(FourierTransform, transform3D){
	//Setup input array.
	Array<std::complex<double>> input({10, 5, 3});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			for(unsigned int z = 0; z < 3; z++)
				input[{x, y, z}] = x*y*z;

	//Create output array.
	Array<std::complex<double>> output({10, 5, 3});

	//Create reference array.
	Array<std::complex<double>> reference;

	for(int sign = -1; sign < 2; sign += 2){
		//Perform FFT.
		FourierTransform::transform(
			input.getData(),
			output.getData(),
			10,
			5,
			3,
			sign
		);

		//Calculate reference solution.
		reference = Array<std::complex<double>>({10, 5, 3}, 0);
		for(unsigned int kx = 0; kx < 10; kx++){
			for(unsigned int ky = 0; ky < 5; ky++){
				for(unsigned int kz = 0; kz < 3; kz++){
					for(unsigned int x = 0; x < 10; x++){
						for(
							unsigned int y = 0;
							y < 5;
							y++
						){
							for(
								unsigned int z = 0;
								z < 3;
								z++
							){
								double KX = 2*M_PI*(kx/10.);
								double KY = 2*M_PI*(ky/5.);
								double KZ = 2*M_PI*(kz/3.);
								reference[{kx, ky, kz}] += input[
									{x, y, z}
								]*exp(
									((double)sign)*i*(
										KX*(double)x
										+ KY*(double)y
										+ KZ*(double)z
									)
								);
							}
						}
					}
				}
			}
		}
		for(unsigned int kx = 0; kx < 10; kx++){
			for(unsigned int ky = 0; ky < 5; ky++){
				for(unsigned int kz = 0; kz < 3; kz++){
					reference[{kx, ky, kz}]
						/= sqrt(10.*5.*3.);
				}
			}
		}

		//Check the output against the reference solution.
		for(unsigned int kx = 0; kx < 10; kx++){
			for(unsigned int ky = 0; ky < 5; ky++){
				for(unsigned int kz = 0; kz < 3; kz++){
					EXPECT_NEAR(
						real(output[{kx, ky, kz}]),
						real(reference[{kx, ky, kz}]),
						EPSILON_10000
					);
					EXPECT_NEAR(
						imag(output[{kx, ky, kz}]),
						imag(reference[{kx, ky, kz}]),
						EPSILON_10000
					);
				}
			}
		}
	}
}

TEST(FourierTransform, transformND){
	//Test against the explicitly 2-dimensional transform that has already
	//been test in the Test FourierTransform::transform2D.

	//Setup input array.
	Array<std::complex<double>> input({10, 5});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			input[{x, y}] = x*y;

	//Create output array.
	Array<std::complex<double>> output({10, 5});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5});

	for(int sign = -1; sign < 2; sign += 2){
		//Perform FFT.
		FourierTransform::transform(
			input.getData(),
			output.getData(),
			{10, 5},
			sign
		);

		//Calculate reference solution.
		FourierTransform::transform(
			input.getData(),
			reference.getData(),
			10,
			5,
			sign
		);

		//Check the output against the reference solution.
		for(unsigned int kx = 0; kx < 10; kx++){
			for(unsigned int ky = 0; ky < 5; ky++){
				EXPECT_NEAR(
					real(output[{kx, ky}]),
					real(reference[{kx, ky}]),
					EPSILON_10000
				);
				EXPECT_NEAR(
					imag(output[{kx, ky}]),
					imag(reference[{kx, ky}]),
					EPSILON_10000
				);
			}
		}
	}
}

TEST(FourierTransform, transformPlan){
	//Test against the explicitly 2-dimensional transform that has already
	//been test in the Test FourierTransform::transform2D.

	//Setup input array.
	Array<std::complex<double>> input({10, 5});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			input[{x, y}] = x*y;

	//Create output array.
	Array<std::complex<double>> output({10, 5});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5});

	for(int sign = -1; sign < 2; sign += 2){
		//Setup plan.
		FourierTransform::Plan<std::complex<double>> plan(
			input.getData(),
			output.getData(),
			{10, 5},
			sign
		);
		plan.setNormalizationFactor(2);

		//Perform FFT.
		FourierTransform::transform(plan);

		//Renormalize the transform to be able to check against the
		//default normalization in the reference solution.
		for(unsigned int kx = 0; kx < 10; kx++)
			for(unsigned int ky = 0; ky < 5; ky++)
				output[{kx, ky}] /= sqrt(10.*5.)/2.;

		//Calculate reference solution.
		FourierTransform::transform(
			input.getData(),
			reference.getData(),
			10,
			5,
			sign
		);

		//Check the output against the reference solution.
		for(unsigned int kx = 0; kx < 10; kx++){
			for(unsigned int ky = 0; ky < 5; ky++){
				EXPECT_NEAR(
					real(output[{kx, ky}]),
					real(reference[{kx, ky}]),
					EPSILON_10000
				);
				EXPECT_NEAR(
					imag(output[{kx, ky}]),
					imag(reference[{kx, ky}]),
					EPSILON_10000
				);
			}
		}
	}
}

TEST(FourierTransform, forward1D){
	//Setup input array.
	Array<std::complex<double>> input({10});
	for(unsigned int x = 0; x < 10; x++)
		input[{x}] = x;

	//Create output array.
	Array<std::complex<double>> output({10});

	//Create reference array.
	Array<std::complex<double>> reference({10});

	//Perform FFT.
	FourierTransform::forward(
		input.getData(),
		output.getData(),
		10
	);

	//Calculate reference solution.
	FourierTransform::transform(
		input.getData(),
		reference.getData(),
		10,
		-1
	);

	//Check the output against the reference solution.
	for(unsigned int k = 0; k < 10; k++){
		EXPECT_NEAR(
			real(output[{k}]),
			real(reference[{k}]),
			EPSILON_10000
		);
		EXPECT_NEAR(
			imag(output[{k}]),
			imag(reference[{k}]),
			EPSILON_10000
		);
	}
}

TEST(FourierTransform, forward2D){
	//Setup input array.
	Array<std::complex<double>> input({10, 5});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			input[{x, y}] = x*y;

	//Create output array.
	Array<std::complex<double>> output({10, 5});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5});

	//Perform FFT.
	FourierTransform::forward(
		input.getData(),
		output.getData(),
		10,
		5
	);

	//Calculate reference solution.
	FourierTransform::transform(
		input.getData(),
		reference.getData(),
		10,
		5,
		-1
	);

	//Check the output against the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++){
		for(unsigned int ky = 0; ky < 5; ky++){
			EXPECT_NEAR(
				real(output[{kx, ky}]),
				real(reference[{kx, ky}]),
				EPSILON_10000
			);
			EXPECT_NEAR(
				imag(output[{kx, ky}]),
				imag(reference[{kx, ky}]),
				EPSILON_10000
			);
		}
	}
}

TEST(FourierTransform, forward3D){
	//Setup input array.
	Array<std::complex<double>> input({10, 5, 3});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			for(unsigned int z = 0; z < 3; z++)
				input[{x, y, z}] = x*y*z;

	//Create output array.
	Array<std::complex<double>> output({10, 5, 3});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5, 3});

	//Perform FFT.
	FourierTransform::forward(
		input.getData(),
		output.getData(),
		10,
		5,
		3
	);

	//Calculate reference solution.
	FourierTransform::transform(
		input.getData(),
		reference.getData(),
		10,
		5,
		3,
		-1
	);

	//Check the output against the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++){
		for(unsigned int ky = 0; ky < 5; ky++){
			for(unsigned int kz = 0; kz < 3; kz++){
				EXPECT_NEAR(
					real(output[{kx, ky, kz}]),
					real(reference[{kx, ky, kz}]),
					EPSILON_10000
				);
				EXPECT_NEAR(
					imag(output[{kx, ky, kz}]),
					imag(reference[{kx, ky, kz}]),
					EPSILON_10000
				);
			}
		}
	}
}

TEST(FourierTransform, forwardND){
	//Test against the explicitly 2-dimensional transform that has already
	//been test in the Test FourierTransform::forward2D.

	//Setup input array.
	Array<std::complex<double>> input({10, 5});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			input[{x, y}] = x*y;

	//Create output array.
	Array<std::complex<double>> output({10, 5});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5});

	//Perform FFT.
	FourierTransform::forward(
		input.getData(),
		output.getData(),
		{10, 5}
	);

	//Calculate reference solution.
	FourierTransform::forward(
		input.getData(),
		reference.getData(),
		10,
		5
	);

	//Check the output against the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++){
		for(unsigned int ky = 0; ky < 5; ky++){
			EXPECT_NEAR(
				real(output[{kx, ky}]),
				real(reference[{kx, ky}]),
				EPSILON_10000
			);
			EXPECT_NEAR(
				imag(output[{kx, ky}]),
				imag(reference[{kx, ky}]),
				EPSILON_10000
			);
		}
	}
}

TEST(FourierTransform, forwardPlan){
	//Test against the explicitly 2-dimensional transform that has already
	//been test in the Test FourierTransform::forward2D.

	//Setup input array.
	Array<std::complex<double>> input({10, 5});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			input[{x, y}] = x*y;

	//Create output array.
	Array<std::complex<double>> output({10, 5});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5});

	//Setup plan.
	FourierTransform::ForwardPlan<std::complex<double>> plan(
		input.getData(),
		output.getData(),
		{10, 5}
	);
	plan.setNormalizationFactor(2);

	//Perform FFT.
	FourierTransform::transform(plan);

	//Renormalize the transform to be able to check against the default
	//normalization in the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++)
		for(unsigned int ky = 0; ky < 5; ky++)
			output[{kx, ky}] /= sqrt(10.*5.)/2.;

	//Calculate reference solution.
	FourierTransform::forward(
		input.getData(),
		reference.getData(),
		10,
		5
	);

	//Check the output against the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++){
		for(unsigned int ky = 0; ky < 5; ky++){
			EXPECT_NEAR(
				real(output[{kx, ky}]),
				real(reference[{kx, ky}]),
				EPSILON_10000
			);
			EXPECT_NEAR(
				imag(output[{kx, ky}]),
				imag(reference[{kx, ky}]),
				EPSILON_10000
			);
		}
	}
}

TEST(FourierTransform, inverse1D){
	//Setup input array.
	Array<std::complex<double>> input({10});
	for(unsigned int x = 0; x < 10; x++)
		input[{x}] = x;

	//Create output array.
	Array<std::complex<double>> output({10});

	//Create reference array.
	Array<std::complex<double>> reference({10});

	//Perform FFT.
	FourierTransform::inverse(
		input.getData(),
		output.getData(),
		10
	);

	//Calculate reference solution.
	FourierTransform::transform(
		input.getData(),
		reference.getData(),
		10,
		1
	);

	//Check the output against the reference solution.
	for(unsigned int k = 0; k < 10; k++){
		EXPECT_NEAR(
			real(output[{k}]),
			real(reference[{k}]),
			EPSILON_10000
		);
		EXPECT_NEAR(
			imag(output[{k}]),
			imag(reference[{k}]),
			EPSILON_10000
		);
	}
}

TEST(FourierTransform, inversed2D){
	//Setup input array.
	Array<std::complex<double>> input({10, 5});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			input[{x, y}] = x*y;

	//Create output array.
	Array<std::complex<double>> output({10, 5});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5});

	//Perform FFT.
	FourierTransform::inverse(
		input.getData(),
		output.getData(),
		10,
		5
	);

	//Calculate reference solution.
	FourierTransform::transform(
		input.getData(),
		reference.getData(),
		10,
		5,
		1
	);

	//Check the output against the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++){
		for(unsigned int ky = 0; ky < 5; ky++){
			EXPECT_NEAR(
				real(output[{kx, ky}]),
				real(reference[{kx, ky}]),
				EPSILON_10000
			);
			EXPECT_NEAR(
				imag(output[{kx, ky}]),
				imag(reference[{kx, ky}]),
				EPSILON_10000
			);
		}
	}
}

TEST(FourierTransform, inverse3D){
	//Setup input array.
	Array<std::complex<double>> input({10, 5, 3});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			for(unsigned int z = 0; z < 3; z++)
				input[{x, y, z}] = x*y*z;

	//Create output array.
	Array<std::complex<double>> output({10, 5, 3});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5, 3});

	//Perform FFT.
	FourierTransform::inverse(
		input.getData(),
		output.getData(),
		10,
		5,
		3
	);

	//Calculate reference solution.
	FourierTransform::transform(
		input.getData(),
		reference.getData(),
		10,
		5,
		3,
		1
	);

	//Check the output against the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++){
		for(unsigned int ky = 0; ky < 5; ky++){
			for(unsigned int kz = 0; kz < 3; kz++){
				EXPECT_NEAR(
					real(output[{kx, ky, kz}]),
					real(reference[{kx, ky, kz}]),
					EPSILON_10000
				);
				EXPECT_NEAR(
					imag(output[{kx, ky, kz}]),
					imag(reference[{kx, ky, kz}]),
					EPSILON_10000
				);
			}
		}
	}
}

TEST(FourierTransform, inverseND){
	//Test against the explicitly 2-dimensional transform that has already
	//been test in the Test FourierTransform::inverse2D.

	//Setup input array.
	Array<std::complex<double>> input({10, 5});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			input[{x, y}] = x*y;

	//Create output array.
	Array<std::complex<double>> output({10, 5});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5});

	//Perform FFT.
	FourierTransform::inverse(
		input.getData(),
		output.getData(),
		{10, 5}
	);

	//Calculate reference solution.
	FourierTransform::inverse(
		input.getData(),
		reference.getData(),
		10,
		5
	);

	//Check the output against the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++){
		for(unsigned int ky = 0; ky < 5; ky++){
			EXPECT_NEAR(
				real(output[{kx, ky}]),
				real(reference[{kx, ky}]),
				EPSILON_10000
			);
			EXPECT_NEAR(
				imag(output[{kx, ky}]),
				imag(reference[{kx, ky}]),
				EPSILON_10000
			);
		}
	}
}

TEST(FourierTransform, inversePlan){
	//Test against the explicitly 2-dimensional transform that has already
	//been test in the Test FourierTransform::inverse2D.

	//Setup input array.
	Array<std::complex<double>> input({10, 5});
	for(unsigned int x = 0; x < 10; x++)
		for(unsigned int y = 0; y < 5; y++)
			input[{x, y}] = x*y;

	//Create output array.
	Array<std::complex<double>> output({10, 5});

	//Create reference array.
	Array<std::complex<double>> reference({10, 5});

	//Setup plan.
	FourierTransform::InversePlan<std::complex<double>> plan(
		input.getData(),
		output.getData(),
		{10, 5}
	);
	plan.setNormalizationFactor(2);

	//Perform FFT.
	FourierTransform::transform(plan);

	//Renormalize the transform to be able to check against the default
	//normalization in the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++)
		for(unsigned int ky = 0; ky < 5; ky++)
			output[{kx, ky}] /= sqrt(10.*5.)/2.;

	//Calculate reference solution.
	FourierTransform::inverse(
		input.getData(),
		reference.getData(),
		10,
		5
	);

	//Check the output against the reference solution.
	for(unsigned int kx = 0; kx < 10; kx++){
		for(unsigned int ky = 0; ky < 5; ky++){
			EXPECT_NEAR(
				real(output[{kx, ky}]),
				real(reference[{kx, ky}]),
				EPSILON_10000
			);
			EXPECT_NEAR(
				imag(output[{kx, ky}]),
				imag(reference[{kx, ky}]),
				EPSILON_10000
			);
		}
	}
}

};	//End of namespace.
