/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file ExactDiagonalizer.h
 *  @brief Solves a Model using exact diagonalization.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_EXACT_DIAGONALIZATION
#define COM_DAFER45_TBTK_SOLVER_EXACT_DIAGONALIZATION

#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/FockStateRuleSet.h"
#include "TBTK/FockStateRule/WrapperRule.h"
#include "TBTK/InteractionAmplitudeSet.h"
#include "TBTK/Model.h"
#include "TBTK/ManyParticleContext.h"
#include "TBTK/Solver/Solver.h"

#include <initializer_list>

namespace TBTK{
namespace Solver{

class ExactDiagonalizer : public Solver{
	TBTK_DYNAMIC_TYPE_INFORMATION(ExactDiagonalizer)
public:
	/** Constructor. */
	ExactDiagonalizer(
//		Model *model
	);

	/** Destructor. */
	virtual ~ExactDiagonalizer();

	/** Add FockStateRule. */
	unsigned int addSubspace(std::initializer_list<const FockStateRule::WrapperRule> rules);

	/** Add FockStateRule. */
	unsigned int addSubspace(std::vector<FockStateRule::WrapperRule> rules);

	/** Add FockStateRule. */
	unsigned int addSubspace(const FockStateRuleSet &rules);

	/** Run calculation. */
	void run(unsigned int subspace);

	/** Get eigen values. */
	const CArray<double>& getEigenValues(unsigned int subspace);

	/** Get eigen value. */
	const double getEigenValue(unsigned int subspace, int state);

	/** Get amplitude for a given eigenvector \f$n\f$ and physical index
	 *  \f$x\f$: \f$\Psi_{n}(x)\f$
	 *  @param subspace Subspace identifier.
	 *  @param state Eigenstate number.
	 *  @param index Physical index \f$x\fx. */
	const std::complex<double> getAmplitude(
		unsigned int subspace,
		int state,
		const Index &index
	);

	/** Get Model. */
//	Model* getModel();
private:
	/** Model to work on. */
//	Model *model;

	/** Subspace context containing rules, a many-body model, and a
	 *  diagonalization solver for a specific subspace. */
	class SubspaceContext{
	public:
		/** Constructor. */
		SubspaceContext(
			std::initializer_list<const FockStateRule::WrapperRule> rules
		);

		/** Constructor. */
		SubspaceContext(
			std::vector<FockStateRule::WrapperRule> rules
		);

		/** Constructor. */
		SubspaceContext(
			const FockStateRuleSet &rules
		);

		/** Destructor. */
		~SubspaceContext();

		/** Subspace rules. */
//		std::vector<FockStateRule::WrapperRule> rules;
		FockStateRuleSet fockStateRuleSet;

		/** Pointer to many-body model. */
//		Model *manyParticleModel;
		std::shared_ptr<Model> manyParticleModel;

		/** Pointer to diagonalization solver. */
//		Diagonalizer *dSolver;
		std::shared_ptr<Diagonalizer> dSolver;
	private:
	};

	/** Subspace contexts. */
	std::vector<SubspaceContext> subspaceContexts;

	/** Setup many-body mapping. */
	void setupManyParticleModel(unsigned int subspace);

	/** Setup many-body model. */
	template<typename BIT_REGISTER>
	void setupManyParticleModel(unsigned int subspace);
};

inline const CArray<double>& ExactDiagonalizer::getEigenValues(
	unsigned int subspace
){
	return subspaceContexts.at(subspace).dSolver->getEigenValues();
}

inline const double ExactDiagonalizer::getEigenValue(
	unsigned int subspace,
	int state
){
	return subspaceContexts.at(subspace).dSolver->getEigenValue(state);
}

inline const std::complex<double> ExactDiagonalizer::getAmplitude(
	unsigned int subspace,
	int state,
	const Index &index
){
	return subspaceContexts.at(subspace).dSolver->getAmplitude(
		state,
		index
	);
}

/*inline Model* ExactDiagonalizer::getModel(){
	return model;
}*/

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
/// @endcond
