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
 *  @file DataManagerFrame.h
 *  @brief Frame for displaying content handeld by a DataManager.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DATA_MANAGER_FRAME
#define COM_DAFER45_TBTK_DATA_MANAGER_FRAME

#include "DataManager.h"
#include "ParameterSliderPanel.h"

#include <wx/wx.h>
#include <wx/sizer.h>

namespace TBTK{

class DataManagerFrame : public wxFrame{
public:
	/** Constructor. */
	DataManagerFrame(wxWindow *parent);

	/** Destructor. */
	~DataManagerFrame();

	/** Set DataManager. */
	void setDataManager(
		const DataManager *dataManager,
		bool isOwner = false
	);

	/** On paint event. */
	void onPaintEvent(wxPaintEvent &event);

	/** On size event. */
	void onSizeEvent(wxSizeEvent &event);

	/** On menu->file->open. */
	void onMenuFileOpen(wxCommandEvent &event);

	/** On menu->file->quit. */
	void onMenuFileQuit(wxCommandEvent &event);

	/** On menu->help->about. */
	void onMenuHelpAbout(wxCommandEvent &event);

	/** On checkbox event. */
	void onCheckboxEvent(wxCommandEvent &event);

	/** On slider change. */
	void onSliderChange(wxScrollEvent &event);
protected:
	DECLARE_EVENT_TABLE();
private:
	/** DataManager. */
	const DataManager *dataManager;

	/** Flag indicating whether the fram is the owner of the DataManager.
	 */
	bool isOwner;

	/** Checkboxes. */
	std::vector<wxCheckBox*> checkboxes;

	/** Parameter sliders. */
	std::vector<ParameterSliderPanel*> parameterSliders;

	/** Data type sizer. */
	wxSizer* dataTypeSizer;

	/** Parameter sizer. */
	wxSizer* parameterSizer;

	/** Current data windows. */
	std::vector<wxWindow*> currentDataWindows;

	/** Update open windows. */
	void updateOpenWindows();
};

};	//End namespace TBTK

#endif
