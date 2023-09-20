#include "TBTK/Solver/BlockDiagonalizer.h"
#include <iostream>
#include "gtest/gtest.h"

namespace TBTK{
namespace Solver{

TEST(BlockDiagonalizer, Constructor){
	//Not testable on its own.
}

TEST(BlockDiagonalizer, Destructor){
	//Not testable on its own.
}

class SelfConsistencyCallback :
	public BlockDiagonalizer::SelfConsistencyCallback
{
public:
	int selfConsistencyCounter;

	bool selfConsistencyCallback(BlockDiagonalizer &diagonalizer){
		selfConsistencyCounter++;
		if(selfConsistencyCounter == 10)
			return true;
		else
			return false;
	}
} selfConsistencyCallback;

TEST(BlockDiagonalizer, setSelfConsistencyCallback){
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(1, {0, 1}, {0, 0}) + HC;
	model.construct();

	for(unsigned int n = 0; n < 2; n++){
		BlockDiagonalizer solver;
		if(n == 0)
			solver.setParallelExecution(false);
		else
			solver.setParallelExecution(true);
		solver.setVerbose(false);
		solver.setModel(model);
		selfConsistencyCallback.selfConsistencyCounter = 0;
		solver.setSelfConsistencyCallback(selfConsistencyCallback);
		solver.run();
		EXPECT_EQ(selfConsistencyCallback.selfConsistencyCounter, 10);

		selfConsistencyCallback.selfConsistencyCounter = 0;
		solver.setSelfConsistencyCallback(selfConsistencyCallback);
		solver.setMaxIterations(5);
		solver.run();
		EXPECT_EQ(selfConsistencyCallback.selfConsistencyCounter, 5);

		/////////////////////////////
		// Test GPU implementation //
		/////////////////////////////
		BlockDiagonalizer solver2;
		solver2.setVerbose(false);
		solver2.setModel(model);
		solver2.setUseGPUAcceleration(true);
		#ifdef TBTK_CUDA_ENABLED
			selfConsistencyCallback.selfConsistencyCounter = 0;
			solver2.setSelfConsistencyCallback(selfConsistencyCallback);
			solver2.run();
			EXPECT_EQ(selfConsistencyCallback.selfConsistencyCounter, 10);

			selfConsistencyCallback.selfConsistencyCounter = 0;
			solver2.setSelfConsistencyCallback(selfConsistencyCallback);
			solver2.setMaxIterations(5);
			solver2.run();
			EXPECT_EQ(selfConsistencyCallback.selfConsistencyCounter, 5);
		#else
			EXPECT_EXIT(
				{
					Streams::setStdMuteErr();
					solver2.run();
				},
				::testing::ExitedWithCode(1),
				""
			);
		#endif
	}
}

TEST(BlockDiagonalizer, setMaxIterations){
	//Tested through Diagonalizer::setSelfConsistencyCallback
}

TEST(BlockDiagonalizer, run){
	//Tested through
	//Diagonalizer::setSelfConsistencyCallback
	//Diagonalizer::getEigenValues
	//Diagonalizer::getEigenVectors
	//Diagonalizer::getEigenValue
	//Diagonalizer::getAmplitude
}

TEST(BlockDiagonalizer, getEigenValue){
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(1, {0, 1}, {0, 0}) + HC;
	model << HoppingAmplitude(2, {1, 0}, {1, 0});
	model << HoppingAmplitude(3, {2, 1}, {2, 0}) + HC;
	model.construct();

	for(unsigned int n = 0; n < 2; n++){
		BlockDiagonalizer solver;
		if(n == 0)
			solver.setParallelExecution(false);
		else
			solver.setParallelExecution(true);
		solver.setVerbose(false);
		solver.setModel(model);
		solver.run();

		//Access using global state index.
		EXPECT_DOUBLE_EQ(solver.getEigenValue(0), -1);
		EXPECT_DOUBLE_EQ(solver.getEigenValue(1), 1);
		EXPECT_DOUBLE_EQ(solver.getEigenValue(2), 2);
		EXPECT_DOUBLE_EQ(solver.getEigenValue(3), -3);
		EXPECT_DOUBLE_EQ(solver.getEigenValue(4), 3);

		//Access using block state index.
		EXPECT_DOUBLE_EQ(solver.getEigenValue({0}, 0), -1);
		EXPECT_DOUBLE_EQ(solver.getEigenValue({0}, 1), 1);
		EXPECT_DOUBLE_EQ(solver.getEigenValue({1}, 0), 2);
		EXPECT_DOUBLE_EQ(solver.getEigenValue({2}, 0), -3);
		EXPECT_DOUBLE_EQ(solver.getEigenValue({2}, 1), 3);
	}
	/////////////////////////////
	// Test GPU implementation //
	/////////////////////////////
	#ifdef TBTK_CUDA_ENABLED
		for(unsigned int n = 0; n < 2; n++){
			BlockDiagonalizer solver;
			solver.setUseGPUAcceleration(true);
			if(n == 0)
				solver.setParallelExecution(false);
			else
				solver.setParallelExecution(true);
			solver.setVerbose(false);
			solver.setModel(model);
			solver.run();
			std::cout << solver.getEigenValue(0) << " != " << -1 << std::endl;
			//Access using global state index.
			EXPECT_DOUBLE_EQ(solver.getEigenValue(0), -1);
			EXPECT_DOUBLE_EQ(solver.getEigenValue(1), 1);
			EXPECT_DOUBLE_EQ(solver.getEigenValue(2), 2);
			EXPECT_DOUBLE_EQ(solver.getEigenValue(3), -3);
			EXPECT_DOUBLE_EQ(solver.getEigenValue(4), 3);

			//Access using block state index.
			EXPECT_DOUBLE_EQ(solver.getEigenValue({0}, 0), -1);
			EXPECT_DOUBLE_EQ(solver.getEigenValue({0}, 1), 1);
			EXPECT_DOUBLE_EQ(solver.getEigenValue({1}, 0), 2);
			EXPECT_DOUBLE_EQ(solver.getEigenValue({2}, 0), -3);
			EXPECT_DOUBLE_EQ(solver.getEigenValue({2}, 1), 3);
		}
	#endif
}

TEST(BlockDiagonalizer, getEigenVectors){
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(1, {0, 1}, {0, 0}) + HC;
	model << HoppingAmplitude(2, {1, 0}, {1, 0});
	model << HoppingAmplitude(3, {2, 1}, {2, 0}) + HC;
	model.construct();

	for(unsigned int n = 0; n < 2; n++){
		BlockDiagonalizer solver;
		if(n == 0)
			solver.setParallelExecution(false);
		else
			solver.setParallelExecution(true);
		solver.setVerbose(false);
		solver.setModel(model);
		solver.run();

		//Access using global state index.
		EXPECT_DOUBLE_EQ(
			real(solver.getAmplitude(0, {0, 0})/solver.getAmplitude(0, {0, 1})),
			-1
		);
		EXPECT_DOUBLE_EQ(
			imag(solver.getAmplitude(0, {0, 0})/solver.getAmplitude(0, {0, 1})),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(solver.getAmplitude(1, {0, 0})/solver.getAmplitude(1, {0, 1})),
			1
		);
		EXPECT_DOUBLE_EQ(
			imag(solver.getAmplitude(1, {0, 0})/solver.getAmplitude(1, {0, 1})),
			0
		);

		EXPECT_DOUBLE_EQ(real(solver.getAmplitude(2, {1, 0})), 1);
		EXPECT_DOUBLE_EQ(imag(solver.getAmplitude(2, {1, 0})), 0);

		EXPECT_DOUBLE_EQ(
			real(solver.getAmplitude(3, {2, 0})/solver.getAmplitude(3, {2, 1})),
			-1
		);
		EXPECT_DOUBLE_EQ(
			imag(solver.getAmplitude(3, {2, 0})/solver.getAmplitude(3, {2, 1})),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(solver.getAmplitude(4, {2, 0})/solver.getAmplitude(4, {2, 1})),
			1
		);
		EXPECT_DOUBLE_EQ(
			imag(solver.getAmplitude(4, {2, 0})/solver.getAmplitude(4, {2, 1})),
			0
		);

		//Give zero for valid Indices that are outside the block that the
		//eigenstate belongs to.
		EXPECT_DOUBLE_EQ(real(solver.getAmplitude(0, {1, 0})), 0);
		EXPECT_DOUBLE_EQ(imag(solver.getAmplitude(0, {1, 0})), 0);
		EXPECT_DOUBLE_EQ(real(solver.getAmplitude(0, {2, 0})), 0);
		EXPECT_DOUBLE_EQ(imag(solver.getAmplitude(0, {2, 0})), 0);
		EXPECT_DOUBLE_EQ(real(solver.getAmplitude(0, {2, 1})), 0);
		EXPECT_DOUBLE_EQ(imag(solver.getAmplitude(0, {2, 1})), 0);

		//Access using block state index.
		EXPECT_DOUBLE_EQ(
			real(solver.getAmplitude({0}, 0, {0})/solver.getAmplitude({0}, 0, {1})),
			-1
		);
		EXPECT_DOUBLE_EQ(
			imag(solver.getAmplitude({0}, 0, {0})/solver.getAmplitude({0}, 0, {1})),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(solver.getAmplitude({0}, 1, {0})/solver.getAmplitude({0}, 1, {1})),
			1
		);
		EXPECT_DOUBLE_EQ(
			imag(solver.getAmplitude({0}, 1, {0})/solver.getAmplitude({0}, 1, {1})),
			0
		);

		EXPECT_DOUBLE_EQ(real(solver.getAmplitude({1}, 0, {0})), 1);
		EXPECT_DOUBLE_EQ(imag(solver.getAmplitude({1}, 0, {0})), 0);

		EXPECT_DOUBLE_EQ(
			real(solver.getAmplitude({2}, 0, {0})/solver.getAmplitude({2}, 0, {1})),
			-1
		);
		EXPECT_DOUBLE_EQ(
			imag(solver.getAmplitude({2}, 0, {0})/solver.getAmplitude({2}, 0, {1})),
			0
		);
		EXPECT_DOUBLE_EQ(
			real(solver.getAmplitude({2}, 1, {0})/solver.getAmplitude({2}, 1, {1})),
			1
		);
		EXPECT_DOUBLE_EQ(
			imag(solver.getAmplitude({2}, 1, {0})/solver.getAmplitude({2}, 1, {1})),
			0
		);

		//Fail to get amplitude for state with invalid state number.
		::testing::FLAGS_gtest_death_test_style = "threadsafe";
		EXPECT_EXIT(
			{
				Streams::setStdMuteErr();
				solver.getAmplitude({0}, -1, {0});
			},
			::testing::ExitedWithCode(1),
			""
		);
		EXPECT_EXIT(
			{
				Streams::setStdMuteErr();
				solver.getAmplitude({0}, 2, {0});
			},
			::testing::ExitedWithCode(1),
			""
		);
		::testing::FLAGS_gtest_death_test_style = "fast";
	}
	/////////////////////////////
	// Test GPU implementation //
	/////////////////////////////
	#ifdef TBTK_CUDA_ENABLED
		for(unsigned int n = 0; n < 2; n++){
			BlockDiagonalizer solver;
			solver.setUseGPUAcceleration(true);
			if(n == 0)
				solver.setParallelExecution(false);
			else
				solver.setParallelExecution(true);
			solver.setVerbose(false);
			solver.setModel(model);
			solver.run();

			//Access using global state index.
			EXPECT_DOUBLE_EQ(
				real(solver.getAmplitude(0, {0, 0})/solver.getAmplitude(0, {0, 1})),
				-1
			);
			EXPECT_DOUBLE_EQ(
				imag(solver.getAmplitude(0, {0, 0})/solver.getAmplitude(0, {0, 1})),
				0
			);
			EXPECT_DOUBLE_EQ(
				real(solver.getAmplitude(1, {0, 0})/solver.getAmplitude(1, {0, 1})),
				1
			);
			EXPECT_DOUBLE_EQ(
				imag(solver.getAmplitude(1, {0, 0})/solver.getAmplitude(1, {0, 1})),
				0
			);

			EXPECT_DOUBLE_EQ(real(solver.getAmplitude(2, {1, 0})), 1);
			EXPECT_DOUBLE_EQ(imag(solver.getAmplitude(2, {1, 0})), 0);

			EXPECT_DOUBLE_EQ(
				real(solver.getAmplitude(3, {2, 0})/solver.getAmplitude(3, {2, 1})),
				-1
			);
			EXPECT_DOUBLE_EQ(
				imag(solver.getAmplitude(3, {2, 0})/solver.getAmplitude(3, {2, 1})),
				0
			);
			EXPECT_DOUBLE_EQ(
				real(solver.getAmplitude(4, {2, 0})/solver.getAmplitude(4, {2, 1})),
				1
			);
			EXPECT_DOUBLE_EQ(
				imag(solver.getAmplitude(4, {2, 0})/solver.getAmplitude(4, {2, 1})),
				0
			);

			//Give zero for valid Indices that are outside the block that the
			//eigenstate belongs to.
			EXPECT_DOUBLE_EQ(real(solver.getAmplitude(0, {1, 0})), 0);
			EXPECT_DOUBLE_EQ(imag(solver.getAmplitude(0, {1, 0})), 0);
			EXPECT_DOUBLE_EQ(real(solver.getAmplitude(0, {2, 0})), 0);
			EXPECT_DOUBLE_EQ(imag(solver.getAmplitude(0, {2, 0})), 0);
			EXPECT_DOUBLE_EQ(real(solver.getAmplitude(0, {2, 1})), 0);
			EXPECT_DOUBLE_EQ(imag(solver.getAmplitude(0, {2, 1})), 0);

			//Access using block state index.
			EXPECT_DOUBLE_EQ(
				real(solver.getAmplitude({0}, 0, {0})/solver.getAmplitude({0}, 0, {1})),
				-1
			);
			EXPECT_DOUBLE_EQ(
				imag(solver.getAmplitude({0}, 0, {0})/solver.getAmplitude({0}, 0, {1})),
				0
			);
			EXPECT_DOUBLE_EQ(
				real(solver.getAmplitude({0}, 1, {0})/solver.getAmplitude({0}, 1, {1})),
				1
			);
			EXPECT_DOUBLE_EQ(
				imag(solver.getAmplitude({0}, 1, {0})/solver.getAmplitude({0}, 1, {1})),
				0
			);

			EXPECT_DOUBLE_EQ(real(solver.getAmplitude({1}, 0, {0})), 1);
			EXPECT_DOUBLE_EQ(imag(solver.getAmplitude({1}, 0, {0})), 0);

			EXPECT_DOUBLE_EQ(
				real(solver.getAmplitude({2}, 0, {0})/solver.getAmplitude({2}, 0, {1})),
				-1
			);
			EXPECT_DOUBLE_EQ(
				imag(solver.getAmplitude({2}, 0, {0})/solver.getAmplitude({2}, 0, {1})),
				0
			);
			EXPECT_DOUBLE_EQ(
				real(solver.getAmplitude({2}, 1, {0})/solver.getAmplitude({2}, 1, {1})),
				1
			);
			EXPECT_DOUBLE_EQ(
				imag(solver.getAmplitude({2}, 1, {0})/solver.getAmplitude({2}, 1, {1})),
				0
			);

			//Fail to get amplitude for state with invalid state number.
			::testing::FLAGS_gtest_death_test_style = "threadsafe";
			EXPECT_EXIT(
				{
					Streams::setStdMuteErr();
					solver.getAmplitude({0}, -1, {0});
				},
				::testing::ExitedWithCode(1),
				""
			);
			EXPECT_EXIT(
				{
					Streams::setStdMuteErr();
					solver.getAmplitude({0}, 2, {0});
				},
				::testing::ExitedWithCode(1),
				""
			);
			::testing::FLAGS_gtest_death_test_style = "fast";
		}
	#endif
}

TEST(BlockDiagonalizer, getFirstStateInBlock){
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(1, {0, 1}, {0, 0}) + HC;
	model << HoppingAmplitude(2, {1, 0}, {1, 0});
	model << HoppingAmplitude(3, {2, 1}, {2, 0}) + HC;
	model.construct();

	for(unsigned int n = 0; n < 2; n++){
		BlockDiagonalizer solver;
		if(n == 0)
			solver.setParallelExecution(false);
		else
			solver.setParallelExecution(true);
		solver.setVerbose(false);
		solver.setModel(model);
		solver.run();

		EXPECT_EQ(solver.getFirstStateInBlock({0, 0}), 0);
		EXPECT_EQ(solver.getFirstStateInBlock({0, 1}), 0);
		EXPECT_EQ(solver.getFirstStateInBlock({1, 0}), 2);
		EXPECT_EQ(solver.getFirstStateInBlock({2, 0}), 3);
		EXPECT_EQ(solver.getFirstStateInBlock({2, 1}), 3);
	}
	/////////////////////////////
	// Test GPU implementation //
	/////////////////////////////
	#ifdef TBTK_CUDA_ENABLED
		for(unsigned int n = 0; n < 2; n++){
			BlockDiagonalizer solver;
			solver.setUseGPUAcceleration(true);
			if(n == 0)
				solver.setParallelExecution(false);
			else
				solver.setParallelExecution(true);
			solver.setVerbose(false);
			solver.setModel(model);
			solver.run();

			EXPECT_EQ(solver.getFirstStateInBlock({0, 0}), 0);
			EXPECT_EQ(solver.getFirstStateInBlock({0, 1}), 0);
			EXPECT_EQ(solver.getFirstStateInBlock({1, 0}), 2);
			EXPECT_EQ(solver.getFirstStateInBlock({2, 0}), 3);
			EXPECT_EQ(solver.getFirstStateInBlock({2, 1}), 3);
		}
	#endif
}

TEST(BlockDiagonalizer, getLastStateInBlock){
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(1, {0, 1}, {0, 0}) + HC;
	model << HoppingAmplitude(2, {1, 0}, {1, 0});
	model << HoppingAmplitude(3, {2, 1}, {2, 0}) + HC;
	model.construct();

	for(unsigned int n = 0; n < 2; n++){
		BlockDiagonalizer solver;
		if(n == 0)
			solver.setParallelExecution(false);
		else
			solver.setParallelExecution(true);
		solver.setVerbose(false);
		solver.setModel(model);
		solver.run();

		EXPECT_EQ(solver.getLastStateInBlock({0, 0}), 1);
		EXPECT_EQ(solver.getLastStateInBlock({0, 1}), 1);
		EXPECT_EQ(solver.getLastStateInBlock({1, 0}), 2);
		EXPECT_EQ(solver.getLastStateInBlock({2, 0}), 4);
		EXPECT_EQ(solver.getLastStateInBlock({2, 1}), 4);
	}
	/////////////////////////////
	// Test GPU implementation //
	/////////////////////////////
	#ifdef TBTK_CUDA_ENABLED
		for(unsigned int n = 0; n < 2; n++){
			BlockDiagonalizer solver;
			solver.setUseGPUAcceleration(true);
			if(n == 0)
				solver.setParallelExecution(false);
			else
				solver.setParallelExecution(true);
			solver.setVerbose(false);
			solver.setModel(model);
			solver.run();

			EXPECT_EQ(solver.getLastStateInBlock({0, 0}), 1);
			EXPECT_EQ(solver.getLastStateInBlock({0, 1}), 1);
			EXPECT_EQ(solver.getLastStateInBlock({1, 0}), 2);
			EXPECT_EQ(solver.getLastStateInBlock({2, 0}), 4);
			EXPECT_EQ(solver.getLastStateInBlock({2, 1}), 4);
		}
	#endif
}

TEST(BlockDiagonalizer, setParallelExecution){
	//Tested through all other implemented tests.
}

};	//End of namespace Solver
};	//End of namespace TBTK
