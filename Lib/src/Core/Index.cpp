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

/** @file Index.cpp
 *  @author Kristofer Björnson
 */

#include "TBTK/Index.h"
#include "TBTK/IndexException.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <vector>

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

Index::Index(const Index &head, const Index &tail){
	indices.reserve(head.getSize() + tail.getSize());
	indices.insert(
		indices.end(),
		head.indices.begin(),
		head.indices.end()
	);
	indices.insert(
		indices.end(),
		tail.indices.begin(),
		tail.indices.end()
	);
}

Index::Index(initializer_list<initializer_list<int>> indexList){
	for(unsigned int n = 0; n < indexList.size(); n++){
		if(n > 0)
			indices.push_back(IDX_SEPARATOR);
		for(unsigned int c = 0; c < (indexList.begin()+n)->size(); c++)
			indices.push_back(*((indexList.begin() + n)->begin() + c));
	}
}

Index::Index(const vector<vector<int>> &indexList){
	for(unsigned int n = 0; n < indexList.size(); n++){
		if(n > 0)
			indices.push_back(IDX_SEPARATOR);
		for(unsigned int c = 0; c < indexList.at(n).size(); c++)
			indices.push_back(indexList.at(n).at(c));
	}
}

Index::Index(initializer_list<Index> indexList){
	for(unsigned int n = 0; n < indexList.size(); n++){
		if(n > 0)
			indices.push_back(IDX_SEPARATOR);
		for(
			unsigned int c = 0;
			c < (indexList.begin() + n)->getSize();
			c++
		){
			indices.push_back((indexList.begin() + n)->at(c));
		}
	}
}

Index::Index(vector<Index> indexList){
	for(unsigned int n = 0; n < indexList.size(); n++){
		if(n > 0)
			indices.push_back(IDX_SEPARATOR);
		for(
			unsigned int c = 0;
			c < (indexList.begin() + n)->getSize();
			c++
		){
			indices.push_back((indexList.begin() + n)->at(c));
		}
	}
}

Index::Index(const string &indexString){
	TBTKExceptionAssert(
		indexString[0] == '{',
		IndexException(
			"Index::Index()",
			TBTKWhere,
			string("Expected '{' while parsing index string,")
			+ " found '" + indexString[0] + "'.",
			"Specify index using the format \"{X, X, X}\"."
		)
	);

	vector<int> indexVector;
	unsigned int numSeparators = 0;
	bool parsingNumeric = false;
	int numeric = 0;
	for(unsigned int n = 1; n < indexString.size(); n++){
		switch(indexString[n]){
		case '*':
			TBTKExceptionAssert(
				indexVector.size() == numSeparators,
				IndexException(
					"Index::Index()",
					TBTKWhere,
					string("Expected ',' while parsing")
					+ " index string, found '*'.",
					string("Specify index using the")
					+ " format \"{X, X, X}\"."
				)
			);
			TBTKExceptionAssert(
				!parsingNumeric,
				IndexException(
					"Index::Index()",
					TBTKWhere,
					string("Found '*' while parsing")
					+ " numeric value.",
					string("Specify index using the")
					+ " format \"{X, X, X}\"."
				)
			);

			indexVector.push_back(IDX_ALL);
			break;
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
		{
			TBTKExceptionAssert(
				indexVector.size() == numSeparators,
				IndexException(
					"Index::Index()",
					TBTKWhere,
					string("Expected ',' or '}' while")
					+ " parsing index string, found '"
					+ indexString[n] + "'.",
					string("Specify index using the")
					+ " format \"{X, X, X}\"."
				)
			);
			const int ASCII_OFFSET = 48;
			numeric = 10*numeric + (int)indexString[n] - ASCII_OFFSET;
			parsingNumeric = true;
			break;
		}
		case ' ':
			if(parsingNumeric){
				TBTKExceptionAssert(
					indexVector.size() == numSeparators,
					IndexException(
						"Index::Index()",
						TBTKWhere,
						string("Expected ',' while")
						+ " parsing index string,"
						+ " found ' '.",
						string("Specify index using")
						+ " the format \"{X, X, X}\"."
					)
				);

				indexVector.push_back(numeric);
				numeric = 0;
				parsingNumeric = false;
			}
			break;
		case ',':
			if(parsingNumeric){
				indexVector.push_back(numeric);
				numeric = 0;
				parsingNumeric = false;
			}
			TBTKExceptionAssert(
				indexVector.size() == numSeparators+1,
				IndexException(
					"Index::Index()",
					TBTKWhere,
					string("Expected numeric or '}' while")
					+ " parsing index string, found ','.",
					string("Specify index using the")
					+ " format \"{X, X, X}\"."
				)
			);
			numSeparators++;
			break;
		case '}':
			if(parsingNumeric){
				TBTKExceptionAssert(
					indexVector.size() == numSeparators,
					IndexException(
						"Index::Index()",
						TBTKWhere,
						string("Expected ',' while")
						+ " parsing index string,"
						+ " found '}'.",
						string("Specify index using")
						+ " the format \"{X, X, X}\"."
					)
				);

				indexVector.push_back(numeric);
				numeric = 0;
				parsingNumeric = false;
			}
			TBTKExceptionAssert(
				n == indexString.size()-1,
				IndexException(
					"Index::Index()",
					TBTKWhere,
					string("Found '}' before end of index")
					+ " string.",
					string("Specify index using the")
					+ " format \"{X, X, X}\"."
				)
			);
//			n = indexString.size();
			break;
		default:
			throw IndexException(
				"Index::Index()",
				TBTKWhere,
				string("Found '") + indexString[n] + "' while"
				+ " parsing the interior of the index.",
				"Specify index using the format \"{X, X, X}\"."
			);
		}
	}

	TBTKExceptionAssert(
		indexString[indexString.size()-1] == '}',
		IndexException(
			"Index::Index()",
			TBTKWhere,
			string("Expected '}' while reading index string,")
			+ " found '" + indexString[indexString.size()-1]
			+ "'.",
			"Specify index using the format \"{X, X, X}\"."
		)
	);

	for(unsigned int n = 0; n < indexVector.size(); n++)
		indices.push_back(indexVector.at(n));
}

Index::Index(const string &serialization, Serializable::Mode mode){
	switch(mode){
	case Serializable::Mode::Debug:
	{
		TBTKAssert(
			Serializable::validate(serialization, "Index", mode),
			"Index::Index()",
			"Unable to parse string as index '" << serialization
			<< "'.",
			""
		);
		string content = Serializable::getContent(
			serialization,
			Serializable::Mode::Debug
		);

		stringstream ss;
		ss.str(content);
		int subindex;
		while((ss >> subindex)){
			indices.push_back(subindex);
			char c;
			TBTKAssert(
				!(ss >> c) || c == ',',
				"Index::Index()",
				"Unable to parse string as index '" << serialization
				<< "'.",
				""
			);
		}
		break;
	}
	case Serializable::Mode::JSON:
	{
		TBTKAssert(
			Serializable::validate(serialization, "Index", mode),
			"Index::Index()",
			"Unable to parse string as index '" << serialization
			<< "'.",
			""
		);

		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			indices = j.at("indices").get<vector<int>>();
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"Index::Index()",
				"Unable to parse string as index '"
				<< serialization << "'.",
				""
			);
		}

		break;
	}
	default:
		TBTKExit(
			"Index::Index()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

bool operator<(const Index &i1, const Index &i2){
	int minNumIndices;
	if(i1.getSize() < i2.getSize())
		minNumIndices = i1.getSize();
	else
		minNumIndices = i2.getSize();

	for(int n = 0; n < minNumIndices; n++){
		if(i1.at(n) == i2.at(n))
			continue;

		if(i1.at(n) < i2.at(n))
			return true;
		else
			return false;
	}

	if(i1.getSize() == i2.getSize())
		return false;

	TBTKExit(
		"operator<(Index &i1, Index &i2)",
		"Comparison between indices of types mutually incompatible with the TreeNode structure.",
		""
	);
}

bool operator>(const Index &i1, const Index &i2){
	int minNumIndices;
	if(i1.getSize() < i2.getSize())
		minNumIndices = i1.getSize();
	else
		minNumIndices = i2.getSize();

	for(int n = 0; n < minNumIndices; n++){
		if(i1.at(n) == i2.at(n))
			continue;

		if(i1.at(n) < i2.at(n))
			return false;
		else
			return true;
	}

	if(i1.getSize() == i2.getSize())
		return false;

	TBTKExit(
		"operator>(Index &i1, Index &i2)",
		"Comparison between indices of types mutually incompatible with the TreeNode structure.",
		""
	);
}

Index Index::getUnitRange(){
	Index unitRange = *this;

	for(unsigned int n = 0; n < getSize(); n++)
		unitRange.at(n) = 1;

	return unitRange;
}

Index Index::getSubIndex(int first, int last) const{
	vector<int> newSubindices;
	for(int n = first; n <= last; n++)
		newSubindices.push_back(indices.at(n));

	return Index(newSubindices);
}

string Index::serialize(Serializable::Mode mode) const{
	switch(mode){
	case Serializable::Mode::Debug:
	{
		stringstream ss;
		ss << "Index(";
		for(unsigned int n = 0; n < indices.size(); n++){
			if(n != 0)
				ss << ",";
			ss << Serializable::serialize(indices.at(n), mode);
		}
		ss << ")";

		return ss.str();
	}
	case Serializable::Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Index";
		j["indices"] = nlohmann::json(indices);

		return j.dump();
	}
	default:
		TBTKExit(
			"Index::serialize()",
			"Only Serializable::Mode::Debug is supported yet.",
			""
		);
	}
}

};
