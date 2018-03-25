/* Copyright 2018 Kristofer Björnson
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
 *  @file HoppingAmplitudeList.h
 *  @brief List of @link HoppingAmplitude HoppingAmplitudes @endlink.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HOPPING_AMPLITUDE_LIST
#define COM_DAFER45_TBTK_HOPPING_AMPLITUDE_LIST

#include "TBTK/HoppingAmplitude.h"
#include "TBTK/Serializable.h"

#include <vector>

namespace TBTK{

/** @brief List of @link HoppingAmplitude HoppingAmplitudes @endlink. */
class HoppingAmplitudeList : public Serializable{
public:
	/** Constructs a HoppingAmplitudeList */
	HoppingAmplitudeList();

	/** Constructor. Constructs the HoppingAmplitudeList from a
	 *  serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the HoppingAmplitudeList.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	HoppingAmplitudeList(
		const std::string &serializeation,
		Serializable::Mode mode
	);

	/** Append HoppingAmplitude to the end of the list.
	 *
	 *  @param hoppingAmplitude HoppingAmplitude to add. */
	void pushBack(const HoppingAmplitude &hoppingAmplitude);

	/** Get size.
	 *
	 *  @return Number of elements in the list. */
	unsigned int getSize() const;

	/** Subscript operator.
	 *
	 *  @param n Element to access.
	 *
	 *  @return THe HoppingAmplitude at position n. */
	const HoppingAmplitude& operator[](unsigned int n) const;

	/** Serialize HoppingAmplitudeList.
	 *
	 *  @param mode Serialization mode to use.
	 *
	 *  @return Serialized string representation of the
	 *  HoppingAmplitudeList. */
	std::string serialize(Serializable::Mode mode) const;

	/** Get size in bytes.
	 *
	 *  @return Memory size required to store the HoppingAmplitudeList. */
	unsigned int getSizeInBytes() const;
private:
	/** List of hopping amplitudes. */
	std::vector<HoppingAmplitude> hoppingAmplitudes;
};

inline void HoppingAmplitudeList::pushBack(const HoppingAmplitude &hoppingAmplitude){
	hoppingAmplitudes.push_back(hoppingAmplitude);
}

inline unsigned int HoppingAmplitudeList::getSize() const{
	return hoppingAmplitudes.size();
}

inline const HoppingAmplitude& HoppingAmplitudeList::operator[](unsigned int n) const{
	return hoppingAmplitudes[n];
}

inline unsigned int HoppingAmplitudeList::getSizeInBytes() const{
	unsigned int sizeInBytes = 0;
	for(size_t n = 0; n < hoppingAmplitudes.size(); n++)
		sizeInBytes += hoppingAmplitudes[n].getSizeInBytes();
	sizeInBytes += (
		hoppingAmplitudes.capacity() - hoppingAmplitudes.size()
	)*sizeof(HoppingAmplitude);
	sizeInBytes += sizeof(HoppingAmplitudeList);

	return sizeInBytes;
}

};	//End of namespace TBTK

#endif
