/* Copyright 2017 Kristofer Björnson
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

/** @package TBTKcalc
 *  @file ArrayState.h
 *  @brief State class with array based overlap evaluation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ARRAY_STATE
#define COM_DAFER45_TBTK_ARRAY_STATE

#include "TBTK/AbstractState.h"
#include "TBTK/DefaultOperator.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <sstream>

namespace TBTK{

class ArrayState : public AbstractState{
public:
	/** Constructor. */
	ArrayState(
		std::initializer_list<unsigned int> resolution
	);

	/** Destructor. */
	virtual ~ArrayState();

	/** Implements AbstracState::clone(). */
	virtual ArrayState* clone() const;

	/** Implements AbstractState::getOverlapWith(). */
	virtual std::complex<double> getOverlap(const AbstractState &bra) const;

	/** Implements AbstractState::getMatrixElementWith(). */
	virtual std::complex<double> getMatrixElement(
		const AbstractState &bra,
		const AbstractOperator &o = DefaultOperator()
	) const;

	/** Set amplitude. */
	void setAmplitude(
		std::complex<double> amplitude,
		std::initializer_list<unsigned int> element
	);

	/** Set amplitude. */
	void setAmplitude(
		std::complex<double> amplitude,
		const std::vector<unsigned int> &element
	);

	/** Set amplitude. */
	void setAmplitude(
		std::complex<double> amplitude,
		const Index &element
	);

	/** Get amplitude. */
	const std::complex<double>& getAmplitude(
		std::initializer_list<unsigned int> element
	) const;

	/** Get amplitude. */
	const std::complex<double>& getAmplitude(
		const std::vector<unsigned int> &element
	) const;

	/** Get amplitude. */
	const std::complex<double>& getAmplitude(
		const Index &element
	) const;
protected:
	/** Returns a vector containing the storage resolution. */
	const std::vector<unsigned int>& getResolution() const;
private:
	class Storage{
	public:
		/** Constructor. */
		Storage(std::initializer_list<unsigned int> resolution);

		/** Destructor. */
		~Storage();

		/** Grab reference (increments the reference counter). */
		void grab();

		/** Release reference. If the function returns true, the caller
		 *  should delete the Storage. */
		bool release();

		/** Set element. */
		void setElement(
			std::complex<double> value,
			std::initializer_list<unsigned int> element
		);

		/** Set element. */
		void setElement(
			std::complex<double> value,
			const std::vector<unsigned int> &element
		);

		/** Set element. */
		void setElement(
			std::complex<double> value,
			const Index &element
		);

		/** Get element. */
		const std::complex<double>& getElement(
			std::initializer_list<unsigned int> element
		) const;

		/** Get element. */
		const std::complex<double>& getElement(
			const std::vector<unsigned int> &element
		) const;

		/** Get element. */
		const std::complex<double>& getElement(
			const Index &element
		) const;

		/** Get resolution. */
		const std::vector<unsigned int>& getResolution() const;
	private:
		/** Reference counter. */
		unsigned int referenceCounter;

		/** Data. */
		std::complex<double> *data;

		/** Data resolution. */
		std::vector<unsigned int> resolution;
	};

	Storage *storage;
};

inline void ArrayState::Storage::grab(){
	referenceCounter++;
}

inline bool ArrayState::Storage::release(){
	referenceCounter--;
	if(referenceCounter == 0)
		return true;
	else
		return false;
}

inline void ArrayState::setAmplitude(
	std::complex<double> amplitude,
	std::initializer_list<unsigned int> element
){
	storage->setElement(amplitude, element);
}

inline void ArrayState::setAmplitude(
	std::complex<double> amplitude,
	const std::vector<unsigned int> &element
){
	storage->setElement(amplitude, element);
}

inline void ArrayState::setAmplitude(
	std::complex<double> amplitude,
	const Index &element
){
	storage->setElement(amplitude, element);
}

inline const std::complex<double>& ArrayState::getAmplitude(
	std::initializer_list<unsigned int> element
) const{
	return storage->getElement(element);
}

inline const std::complex<double>& ArrayState::getAmplitude(
	const std::vector<unsigned int> &element
) const{
	return storage->getElement(element);
}

inline const std::complex<double>& ArrayState::getAmplitude(
	const Index &element
) const{
	return storage->getElement(element);
}

inline const std::vector<unsigned int>& ArrayState::getResolution() const{
	return storage->getResolution();
}

inline void ArrayState::Storage::setElement(
	std::complex<double> value,
	std::initializer_list<unsigned int> element
){
	unsigned int x = *(element.begin() + 0);
	unsigned int y = *(element.begin() + 1);
	unsigned int z = *(element.begin() + 2);
	data[resolution[2]*(resolution[1]*x + y) + z] = value;
}

inline void ArrayState::Storage::setElement(
	std::complex<double> value,
	const std::vector<unsigned int> &element
){
	unsigned int x = element.at(0);
	unsigned int y = element.at(1);
	unsigned int z = element.at(2);
	data[resolution[2]*(resolution[1]*x + y) + z] = value;
}

inline void ArrayState::Storage::setElement(
	std::complex<double> value,
	const Index &element
){
	unsigned int x = element.at(0);
	unsigned int y = element.at(1);
	unsigned int z = element.at(2);
	data[resolution[2]*(resolution[1]*x + y) + z] = value;
}

inline const std::complex<double>& ArrayState::Storage::getElement(
	std::initializer_list<unsigned int> element
) const{
	unsigned int x = *(element.begin() + 0);
	unsigned int y = *(element.begin() + 1);
	unsigned int z = *(element.begin() + 2);
	return data[resolution[2]*(resolution[1]*x + y) + z];
}

inline const std::complex<double>& ArrayState::Storage::getElement(
	const std::vector<unsigned int> &element
) const{
	unsigned int x = element.at(0);
	unsigned int y = element.at(1);
	unsigned int z = element.at(2);
	return data[resolution[2]*(resolution[1]*x + y) + z];
}

inline const std::complex<double>& ArrayState::Storage::getElement(
	const Index &element
) const{
	unsigned int x = element.at(0);
	unsigned int y = element.at(1);
	unsigned int z = element.at(2);
	return data[resolution[2]*(resolution[1]*x + y) + z];
}

inline const std::vector<unsigned int>& ArrayState::Storage::getResolution() const{
	return resolution;
}

};	//End of namespace TBTK

#endif
