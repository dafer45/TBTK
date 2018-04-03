#include "gtest/gtest.h"

#include "TBTK/Test/BackwardDifference.h"
#include "TBTK/Test/CenteredDifference.h"
#include "TBTK/Test/ForwardDifference.h"
#include "TBTK/Test/HoppingAmplitude.h"
#include "TBTK/Test/HoppingAmplitudeList.h"
#include "TBTK/Test/HoppingAmplitudeSet.h"
#include "TBTK/Test/HoppingAmplitudeTree.h"
#include "TBTK/Test/Index.h"
#include "TBTK/Test/IndexedDataTree.h"
#include "TBTK/Test/Model.h"
#include "TBTK/Test/SingleParticleContext.h"
#include "TBTK/Test/SourceAmplitude.h"
#include "TBTK/Test/SourceAmplitudeSet.h"

int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}
