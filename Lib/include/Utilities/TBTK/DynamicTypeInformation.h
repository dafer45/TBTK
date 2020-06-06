/* Copyright 2020 Kristofer Björnson
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
 *  @file DynamicTypeInformation.h
 *  @brief Enables dynamic type information.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DYNAMIC_TYPE_INFORMATION
#define COM_DAFER45_TBTK_DYNAMIC_TYPE_INFORMATION

#include <string>
#include <vector>

namespace TBTK{

/** @brief Enables dynamic type information. */
class DynamicTypeInformation{
public:
	/** Constructor.
	 *
	 *  @param name The name of the type. Should be equal to the class
	 *  name.
	 *
	 *  @param parents List of parents. Should contain a list of pointers
	 *  to the DynamicTypeInformation in the parent classes that implements
	 *  the have DynamicTypeInformation. */
	DynamicTypeInformation(
		const std::string &name,
		const std::vector<DynamicTypeInformation*> &parents
	);

	/** Get the type name.
	 *
	 *  @return The name of the type. */
	const std::string& getName() const;

	/** Get the number of parents.
	 *
	 *  @return Get the number of immediate parents. */
	unsigned int getNumParents() const;

	/** Get the nth parent.
	 *
	 *  @param n The index of the parent to get. */
	const DynamicTypeInformation& getParent(unsigned int n) const;
private:
	/** Type name. */
	std::string name;

	/** List of parents. */
	std::vector<DynamicTypeInformation*> parents;
};

inline const std::string& DynamicTypeInformation::getName() const{
	return name;
}

inline unsigned int DynamicTypeInformation::getNumParents() const{
	return parents.size();
}

inline const DynamicTypeInformation& DynamicTypeInformation::getParent(
	unsigned int n
) const{
	return *parents[n];
}

#define TBTK_DYNAMIC_TYPE_INFORMATION(type) \
public: \
	/** Dynamic type information. */ \
	static DynamicTypeInformation dynamicTypeInformation; \
	/** Get the DynamicTypeInformation. \
	 * \
	 *  @return Returns the DynamicTypeInformation associated with the \
	 *  class. */ \
	virtual const DynamicTypeInformation& getDynamicTypeInformation( \
	) const{ \
		return dynamicTypeInformation; \
	} \
	/** Get the type name returned by typeid(DataType).name(). \
	 * \
	 *  @return The type name returned by typeid(DataType).name(). */ \
	static std::string getTypeidName(){ \
		return typeid(type).name(); \
	}
/*#define TBTK_DYNAMIC_TYPE_INFORMATION \
public: \
	static DynamicTypeInformation dynamicTypeInformation;*/

};	//End of namespace TBTK

#endif
