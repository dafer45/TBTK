#include "TBTK/PropertyExtractor/PatternValidator.h"

#include "gtest/gtest.h"

#include <complex>

namespace TBTK{
namespace PropertyExtractor{

//TBTKFeature PropertyExtractor.PatternValidator.chechNumRequiredComponentIndices.1 2019-11-01
TEST(PatternValidator, checkNumRequiredComponentIndices1){
	PatternValidator patternValidator;
	patternValidator.validate({
		{1, 2},
		{{1, 2}, {1, 2}}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechNumRequiredComponentIndices.2 2019-11-01
TEST(PatternValidator, checkNumRequiredComponentIndices2){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(2);
	patternValidator.validate({
		{{1, 2}, {1, 2}}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechNumRequiredComponentIndices.3 2019-11-01
TEST(PatternValidator, checkNumRequiredComponentIndices3){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(2);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			patternValidator.validate({
				{1, 2}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.1 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags1){
	PatternValidator patternValidator;
	patternValidator.validate({
		{1, 2}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.2 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags2){
	PatternValidator patternValidator;
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			patternValidator.validate({
				{1, IDX_ALL}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.3 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags3){
	PatternValidator patternValidator;
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SPIN});
	patternValidator.validate({
		{1, 2}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.4 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags4){
	PatternValidator patternValidator;
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SPIN});
	patternValidator.validate({
		{IDX_ALL, IDX_SPIN}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechAllowedSubindexFlags.5 2019-11-01
TEST(PatternValidator, checkAllowedSubindexFlags5){
	PatternValidator patternValidator;
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SPIN});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			patternValidator.validate({
				{IDX_ALL, IDX_SUM_ALL}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.chechRequiredSubindexFlags.0 2020-07-04
TEST(PatternValidator, checkRequiredSubindexFlags0){
	PatternValidator patternValidator;
	patternValidator.setRequiredSubindexFlags({{IDX_SPIN, 1}, {IDX_ALL, 2}});
	patternValidator.validate({
		{1, IDX_SPIN, IDX_ALL, 4, IDX_ALL}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechRequiredSubindexFlags.1 2020-07-04
TEST(PatternValidator, checkRequiredSubindexFlags1){
	PatternValidator patternValidator;
	patternValidator.setRequiredSubindexFlags({{IDX_SPIN, 1}, {IDX_ALL, 2}});
	patternValidator.validate({
		{
			{1, IDX_SPIN, IDX_ALL, 4, IDX_ALL},
			{1, IDX_SPIN, IDX_ALL, 4, IDX_ALL}
		}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechRequiredSubindexFlags.2 2020-07-04
TEST(PatternValidator, checkRequiredSubindexFlags2){
	PatternValidator patternValidator;
	patternValidator.setRequiredSubindexFlags({{IDX_SPIN, 0}});
	patternValidator.validate({{1, IDX_SPIN}});
	patternValidator.validate({{1, IDX_SPIN, IDX_SPIN}});
}

//TBTKFeature PropertyExtractor.PatternValidator.chechRequiredSubindexFlags.3 2020-07-04
TEST(PatternValidator, checkRequiredSubindexFlags3){
	PatternValidator patternValidator;
	patternValidator.setRequiredSubindexFlags({
		{IDX_SPIN, 1}, {IDX_ALL, 2}
	});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			patternValidator.validate({
				{1, IDX_SPIN, IDX_ALL, 4}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.chechRequiredSubindexFlags.4 2020-07-04
TEST(PatternValidator, checkRequiredSubindexFlags4){
	PatternValidator patternValidator;
	patternValidator.setRequiredSubindexFlags({{IDX_SPIN, 1}, {IDX_ALL, 2}});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			patternValidator.validate({
				{
					{1, IDX_SPIN, IDX_ALL, 4, IDX_ALL},
					{1, IDX_SPIN, IDX_ALL, 4}
				}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.chechRequiredSubindexFlags.5 2020-07-04
TEST(PatternValidator, checkRequiredSubindexFlags5){
	PatternValidator patternValidator;
	patternValidator.setRequiredSubindexFlags({{IDX_SPIN, 0}});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			patternValidator.validate({{1, 2}});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateWaveFunctionPatterns.0 2020-07-04
TEST(PatternValidator, validateWaveFunctionPatterns0){
	PatternValidator::validateWaveFunctionPatterns({
		{4, IDX_ALL, IDX_SUM_ALL, 3}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.validateWaveFunctionPatterns.1 2020-07-04
TEST(PatternValidator, validateWaveFunctionPatterns1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateWaveFunctionPatterns({
				{4, IDX_SPIN, IDX_SUM_ALL, 3}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateWaveFunctionPatterns.2 2020-07-04
TEST(PatternValidator, validateWaveFunctionPatterns2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateWaveFunctionPatterns({
				{{4, 3}, {2, 3}}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateGreensFunctionPatterns.0 2020-07-04
TEST(PatternValidator, validateGreensFunctionPatterns0){
	PatternValidator::validateGreensFunctionPatterns({
		{{IDX_ALL, 3}, {2, IDX_SUM_ALL}}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.validateGreensFunctionPatterns.1 2020-07-04
TEST(PatternValidator, validateGreensFunctionPatterns1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateGreensFunctionPatterns({
				{{IDX_SPIN, 3}, {2, IDX_SUM_ALL}}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateGreensFunctionPatterns.2 2020-07-04
TEST(PatternValidator, validateGreensFunctionPatterns2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateGreensFunctionPatterns({
				{4, 3}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateDensityPatterns.0 2020-07-04
TEST(PatternValidator, validateDensityPatterns0){
	PatternValidator::validateDensityPatterns({
		{4, IDX_ALL, IDX_SUM_ALL, 3}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.validateDensityPatterns.1 2020-07-04
TEST(PatternValidator, validateDensityPatterns1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateDensityPatterns({
				{4, IDX_SPIN, IDX_SUM_ALL, 3}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateDensityPatterns.2 2020-07-04
TEST(PatternValidator, validateDensityPatterns2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateDensityPatterns({
				{{4, 3}, {2, 3}}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateMagnetizationPatterns.0 2020-07-04
TEST(PatternValidator, validateMagnetizationPatterns0){
	PatternValidator::validateMagnetizationPatterns({
		{4, IDX_ALL, IDX_SUM_ALL, IDX_SPIN, 3}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.validateMagnetizationPatterns.1 2020-07-04
TEST(PatternValidator, validateMagnetizationPatterns1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateMagnetizationPatterns({
				{{4, 3}, {2, 3}}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateMagnetizationPatterns.2 2020-07-04
TEST(PatternValidator, validateMagnetizationPatterns2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateMagnetizationPatterns({
				{4, IDX_ALL, IDX_SUM_ALL, 3}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateMagnetizationPatterns.3 2020-07-04
TEST(PatternValidator, validateMagnetizationPatterns3){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateMagnetizationPatterns({
				{4, IDX_ALL, IDX_SUM_ALL, IDX_SPIN, IDX_SPIN}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateLDOSPatterns.0 2020-07-04
TEST(PatternValidator, validateLDOSPatterns0){
	PatternValidator::validateLDOSPatterns({
		{4, IDX_ALL, IDX_SUM_ALL, 3}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.validateLDOSPatterns.1 2020-07-04
TEST(PatternValidator, validateLDOSPatterns1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateLDOSPatterns({
				{4, IDX_SPIN, IDX_SUM_ALL, 3}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateLDOSPatterns.2 2020-07-04
TEST(PatternValidator, validateLDOSPatterns2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateLDOSPatterns({
				{{4, 3}, {2, 3}}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateSpinPolarizedLDOSPatterns.0 2020-07-04
TEST(PatternValidator, validateSpinPolarizedLDOSPatterns0){
	PatternValidator::validateSpinPolarizedLDOSPatterns({
		{4, IDX_ALL, IDX_SUM_ALL, IDX_SPIN, 3}
	});
}

//TBTKFeature PropertyExtractor.PatternValidator.validateSpinPolarizedLDOSPatterns.1 2020-07-04
TEST(PatternValidator, validateSpinPolarizedLDOSPatterns1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateSpinPolarizedLDOSPatterns({
				{{4, 3}, {2, 3}}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateSpinPolarizedLDOSPatterns.2 2020-07-04
TEST(PatternValidator, validateSpinPolarizedLDOSPatterns2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateSpinPolarizedLDOSPatterns({
				{4, IDX_ALL, IDX_SUM_ALL, 3}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature PropertyExtractor.PatternValidator.validateSpinPolarizedLDOSPatterns.3 2020-07-04
TEST(PatternValidator, validateSpinPolarizedLDOSPatterns3){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PatternValidator::validateSpinPolarizedLDOSPatterns({
				{4, IDX_ALL, IDX_SUM_ALL, IDX_SPIN, IDX_SPIN}
			});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
