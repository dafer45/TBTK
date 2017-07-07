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
 *  @file IndexPanel.h
 *  @brief Panel for displaying an Index.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INDEX_PANEL
#define COM_DAFER45_TBTK_INDEX_PANEL

#include "Index.h"

#include <wx/wx.h>
#include <wx/sizer.h>

namespace TBTK{

class IndexPanel : public wxPanel{
public:
	/** Constructor. */
	IndexPanel(wxWindow *parent, wxWindowID id = wxID_ANY);

	/** Destructor. */
	~IndexPanel();

	/** Set Index. */
	void setIndex(const Index &index);

	/** Get Index. */
	const Index& getIndex() const;

	/** On paint event. */
	void onPaintEvent(wxPaintEvent &event);

	/** On size event. */
	void onSizeEvent(wxSizeEvent &event);

	/** On Index changed. */
	void onIndexChanged(wxCommandEvent &event);

	/** Redraw. */
	void redraw();
protected:
	DECLARE_EVENT_TABLE();
private:
	/** The Index. */
	Index index;

	/** Index label. */
	wxStaticText indexLabel;

	/** Text box for index. */
	wxTextCtrl indexTextBox;

	static const wxWindowID INDEX_ID;
};

inline const Index& IndexPanel::getIndex() const{
	return index;
}

};	//End namespace TBTK

#endif
