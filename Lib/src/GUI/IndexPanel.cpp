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

/** @file IndexPanel.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/IndexException.h"
#include "TBTK/IndexPanel.h"

#include <wx/gbsizer.h>

using namespace std;

namespace TBTK{

const wxWindowID IndexPanel::INDEX_ID = wxWindow::NewControlId();

IndexPanel::IndexPanel(
	wxWindow *parent,
	wxWindowID id
) :
	wxPanel(parent, id),
	indexLabel(
		this,
		wxID_ANY,
		"Index"
	),
	indexTextBox(
		this,
		INDEX_ID,
		"{}",
		wxDefaultPosition,
		wxDefaultSize,
		wxTE_PROCESS_ENTER | wxTE_PROCESS_TAB,
		wxDefaultValidator,
		wxTextCtrlNameStr
	)
{
	wxSizer *sizer = new wxBoxSizer(wxVERTICAL);
	sizer->Add(&indexLabel, 1, wxEXPAND);
	sizer->Add(&indexTextBox, 1, wxEXPAND);
	SetSizer(sizer);
}

IndexPanel::~IndexPanel(){
}

void IndexPanel::setIndex(const Index &index){
	this->index = index;
	Refresh();
}

BEGIN_EVENT_TABLE(IndexPanel, wxPanel)

EVT_PAINT(IndexPanel::onPaintEvent)
EVT_SIZE(IndexPanel::onSizeEvent)
EVT_TEXT_ENTER(INDEX_ID, IndexPanel::onIndexChanged)
EVT_TEXT(INDEX_ID, IndexPanel::onIndexChanged)

END_EVENT_TABLE()

void IndexPanel::onPaintEvent(wxPaintEvent &event){
}

void IndexPanel::onSizeEvent(wxSizeEvent &event){
	Layout();
	Refresh();
	event.Skip();
}

void IndexPanel::onIndexChanged(wxCommandEvent &event){
	try{
		index = Index(string(event.GetString()));
	}
	catch(IndexException &e){
	}
	event.Skip();
}

};	//End of namespace TBTK
