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

/** @package TBTKcalc
 *  @file AbstractOperator.h
 *  @brief Abstract operator class from which other operators inherit.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ABSTRACT_OPERATOR
#define COM_DAFER45_TBTK_ABSTRACT_OPERATOR

namespace TBTK{

//class AbstractState;

class AbstractOperator{
public:
	/** Destructor. */
	virtual ~AbstractOperator();

	/** List of operator identifiers. Officially supported operators are
	 *  given unique identifiers. Operators not (yet) supported should make
	 *  sure they use an identifier that does not clash with the officially
	 *  supported ones [ideally a large random looking number (magic
	 *  number) to also minimize accidental clashes with other operators
	 *  that are not (yet) supported]. */
	enum OperatorID{
		Default = 0
	};

	/** Get operator identifier. */
	OperatorID getOperatorID() const;
protected:
	/** Constructor. */
	AbstractOperator(OperatorID operatorID);
private:
	/** Operator identifier. */
	OperatorID operatorID;
};

inline AbstractOperator::OperatorID AbstractOperator::getOperatorID() const{
	return operatorID;
}

};	//End of namespace TBTK

#endif
