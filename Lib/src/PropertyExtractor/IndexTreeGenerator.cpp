/* Copyright 2019 Kristofer Björnson
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

/** @file IndexTreeGenerator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/IndexTreeGenerator.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace PropertyExtractor{

IndexTreeGenerator::IndexTreeGenerator(const Model &model) : model(model){
	keepSummationWildcards = false;
	keepSpinWildcards = false;
}

IndexTree IndexTreeGenerator::generate(const vector<Index> &patterns) const{
	IndexTree indexTree;
	const HoppingAmplitudeSet &hoppingAmplitudeSet
		= model.getHoppingAmplitudeSet();

	for(unsigned int n = 0; n < patterns.size(); n++){
		Index pattern = *(patterns.begin() + n);

		vector<Index> components = pattern.split();

		switch(components.size()){
		case 1:
		{
			pattern = components[0];

			for(unsigned int c = 0; c < pattern.getSize(); c++){
				switch(pattern.at(c)){
				case IDX_ALL:
				case IDX_SUM_ALL:
				case IDX_SPIN:
					pattern.at(c) = IDX_ALL;
					break;
				default:
					TBTKAssert(
						pattern.at(c) >= 0,
						"PropertyExtractor::generateIndexTree()",
						"Subindex " << c << " of pattern " << n << " is invalid.",
						"Must be non-negative, IDX_ALL, IDX_SUM_ALL, or IDX_SPIN."
					);
					break;
				}
				if(pattern.at(c) < 0)
					pattern.at(c) = IDX_ALL;
			}

			vector<Index> indices = hoppingAmplitudeSet.getIndexList(
				pattern
			);
			Index p = *(patterns.begin() + n);
			for(unsigned int c = 0; c < indices.size(); c++){
				for(unsigned int m = 0; m < p.getSize(); m++){
					if(
						keepSummationWildcards
						&& p.at(m).isSummationIndex()
					){
						indices.at(c).at(m) = IDX_SUM_ALL;
					}
					if(
						keepSpinWildcards
						&& p.at(m).isSpinIndex()
					){
						indices.at(c).at(m) = IDX_SPIN;
					}
				}
			}
			for(unsigned int c = 0; c < indices.size(); c++)
				indexTree.add(indices.at(c));

			break;
		}
		case 2:
		{
			IndexTree firstIndexTree = generate({components[0]});
			IndexTree secondIndexTree = generate({components[1]});
			for(
				IndexTree::ConstIterator iterator0
					= firstIndexTree.cbegin();
				iterator0 != firstIndexTree.cend();
				++iterator0
			){
				for(
					IndexTree::ConstIterator iterator1
						= secondIndexTree.cbegin();
					iterator1 != secondIndexTree.cend();
					++iterator1
				){
					indexTree.add({*iterator0, *iterator1});
				}
			}

			break;
		}
		default:
			TBTKExit(
				"PropertyExtractor::generateIndexTree()",
				"Only patterns with one and two component"
				<< " Indices are supported so far, but the"
				<< " pattern '" << pattern.toString() << "'"
				<< " has '" << components.size() << "'"
				<< " components.",
				""
			);
		}
	}

	indexTree.generateLinearMap();

	return indexTree;
}

};	//End of namesapce PropertyExtractor
};	//End of namespace TBTK
