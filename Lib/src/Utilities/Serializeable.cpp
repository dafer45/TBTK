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

/** @file Serializeable.cpp
 *
 *  @author Kristofer Björnson
 */

#include "Serializeable.h"
#include "TBTKMacros.h"

#include "json.hpp"

using namespace std;
using namespace nlohmann;

namespace TBTK{

bool Serializeable::validate(
	const string &serialization,
	const std::string &id,
	Mode mode
){
	switch(mode){
	case Mode::Debug:
	{
		size_t position = serialization.find(id + "(");
		if(position != 0 || serialization.back() != ')')
			return false;
/*		TBTKAssert(
			position == 0,
			"Serializeable::validate()",
			"Unable to parse serialization string '"
			<< serialization << "'.",
			""
		);
		TBTKAssert(
			serialization.back() == ')',
			"Serializeable::validate()",
			"Unable to parse serialization string '"
			<< serialization << "'.",
			""
		);*/

		return true;
	}
	case Mode::JSON:
	{
		try{
			json j = json::parse(serialization);
			if(j.at("id").get<string>().compare(id) == 0)
				return true;
			else
				return false;
		}
		catch(json::exception e){
			return false;
		}
	}
	default:
		TBTKExit(
			"Serializeable::validate()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

bool Serializeable::hasID(const string &serialization, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		size_t position = serialization.find('(');
		if(position == string::npos)
			return false;
		else
			return true;
	}
	default:
		TBTKExit(
			"Serializeable::hasID()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

string Serializeable::getID(const string &serialization, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		size_t position = serialization.find('(');
		TBTKAssert(
			position != string::npos,
			"Serializeable::getID()",
			"Unable to parse serialization string '"
			<< serialization << "'.",
			""
		);

		return serialization.substr(0, position);
	}
	default:
		TBTKExit(
			"Serializeable::getID()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

string Serializeable::getContent(const string &serialization, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		size_t position = serialization.find('(');
		TBTKAssert(
			position != string::npos,
			"Serializeable::getContent()",
			"Unable to parse serialization string '"
			<< serialization << "'.",
			""
		);
		TBTKAssert(
			serialization.back() == ')',
			"Serializeable::getContent()",
			"Unable to parse serialization string '"
			<< serialization << "'.",
			""
		);

		size_t contentStart = position + 1;
		size_t contentLength = serialization.size() - contentStart - 1;

		return serialization.substr(contentStart, contentLength);
	}
	default:
		TBTKExit(
			"Serializeable::getContent()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

vector<string> Serializeable::split(const string &content, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		vector<string> result;
		result.push_back(string());
		unsigned int bracketCounter = 0;
		for(unsigned int n = 0; n < content.size(); n++){
			char c = content.at(n);
			if(c == '(')
				bracketCounter++;
			if(c == ')'){
				TBTKAssert(
					bracketCounter > 0,
					"Serializeable::split()",
					"Unable to split malformated serialization"
					<< " conent string '" << content << "'.",
					"Unbalanced brackets."
				);

				bracketCounter--;
			}

			if(c == ',' && bracketCounter == 0)
				result.push_back(string());
			else
				result.back() += c;
		}

		TBTKAssert(
			bracketCounter == 0,
			"Serializeable::split()",
			"Unable to split malformated serialization conent string '"
			<< content << "'.",
			"Unbalanced brackets."
		);

		return result;
	}
	default:
		TBTKExit(
			"Serializeable::split()",
			"Only Serializeable::Mode::Debug is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
