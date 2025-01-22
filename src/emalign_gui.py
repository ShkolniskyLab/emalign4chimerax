from chimerax.core import tools
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import vertical_layout, button_row, ModelMenuButton, CollapsiblePanel, EntriesRow, radio_buttons
from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel
from chimerax.map import Volume
from .emalign_cmd import emalign


class EMalignDialog(ToolInstance):
    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        layout = vertical_layout(parent, margins=(5, 0, 0, 0))

        self.log = session.logger

        # Make menus to choose maps for alignment:
        mf = self._create_emalign_map_menu(parent)
        layout.addWidget(mf)

        # Align button:
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        # Make optional arguments fields:
        options = self._create_options_gui(parent)
        layout.addWidget(options)

        # Make actions guide:
        guide = self._create_guide(parent)
        layout.addWidget(guide)

        # Status line:
        self._status_label = sl = QLabel(parent)
        layout.addWidget(sl)

        layout.addStretch(1)  # Extra space at end

        tw.manage(placement="side")

    def _create_emalign_map_menu(self, parent):
        mf = QFrame(parent)
        mlayout = QHBoxLayout(mf)
        mlayout.setContentsMargins(0, 0, 0, 0)
        mlayout.setSpacing(10)

        fl = QLabel('Align query map', mf)
        mlayout.addWidget(fl)

        self._query_map_menu = qm = ModelMenuButton(self.session, class_filter=Volume)
        vlist = self.session.models.list(type=Volume)
        if vlist:
            qm.value = vlist[0]
        qm.value_changed.connect(self._object_chosen)
        mlayout.addWidget(qm)

        iml = QLabel('to reference map', mf)
        mlayout.addWidget(iml)

        self._ref_map_menu = rm = ModelMenuButton(self.session, class_filter=Volume)
        mlayout.addWidget(rm)
        if vlist:
            rm.value = vlist[0]

        rm.value_changed.connect(self._object_chosen)

        mlayout.addStretch(1)  # extra space at end

        return mf

    def _create_action_buttons(self, parent):
        f, buttons = button_row(parent, [('align', self._emalign),
                                         ('options', self._show_or_hide_options),
                                         ('help', self._show_or_hide_guide)], spacing=10, button_list=True)

        return f

    def _create_guide(self, parent):
        self._guide_panel = g = CollapsiblePanel(parent, title=None)
        f = g.content_area
        EntriesRow(f, ' ')
        EntriesRow(f, 'Downsample - dimension to downsample input volumes to speed up computations. ')
        EntriesRow(f, 'Projections - number of projections to use for alignment. ')
        EntriesRow(f, 'Masking - using only center 90% of the volumes energy to calculate the alignment. ')
        EntriesRow(f, '* the alignment may take a few minutes, don\'t click the screen while EMalign is running.')

        return g

    def _show_or_hide_guide(self):
        self._guide_panel.toggle_panel_display()

    def _create_options_gui(self, parent):
        self._options_panel = p = CollapsiblePanel(parent, title=None)
        f = p.content_area

        # Create downsample option:
        header_dns = EntriesRow(f, 'Downsample:')
        s_real = EntriesRow(f, False, 'None (use actual size)')
        s_64 = EntriesRow(f, True, '64')
        s_128 = EntriesRow(f, False, '128')
        s_256 = EntriesRow(f, False, '256')
        self._no_downsample, self._downsample_64, self._downsample_128, self._downsample_256 = s_real.values[0], s_64.values[0], s_128.values[0], s_256.values[0]
        radio_buttons(self._no_downsample, self._downsample_64, self._downsample_128, self._downsample_256)
        self._ds_frames = s_real.frame, s_64.frame, s_128.frame, s_256.frame
        self._no_ds_frame, self._ds_64_frame, self._ds_128_frame, self._ds_256_frame = self._ds_frames
        self._ds_header_frame = header_dns.frame
        frames_ds = [self._ds_header_frame, self._no_ds_frame, self._ds_64_frame, self._ds_128_frame, self._ds_256_frame]

        # Create projections option:
        header_proj = EntriesRow(f, 'Projections:')
        per = EntriesRow(f, True, '25 (default)', False, '50', False, '125 (for noisier data)')
        self._projections_25, self._projections_50, self._projections_125 = per.values
        radio_buttons(self._projections_25, self._projections_50, self._projections_125)
        self._projections_frame = per.frame
        self.header_proj_frame = header_proj.frame

        # Create refinement with 'Fit in Map' option:
        use_fit_map = EntriesRow(f, True, 'Use Fit in Map for final refinement (recommended)')
        self._use_fit_map = use_fit_map.values[0]
        self._use_fit_map_frame = use_fit_map.frame

        # Create option to display detailed log:
        log = EntriesRow(f, False, 'Display detailed log')
        self._display_log = log.values[0]
        self._display_log_frame = log.frame

        # Create option to use masking before alignning:
        mask = EntriesRow(f, False, 'Use masking (recommended only for noisier data)')
        self._masking = mask.values[0]
        self._masking_frame = mask.frame

        additional_frames = [self.header_proj_frame, self._projections_frame, self._display_log_frame, self._masking_frame, self._use_fit_map_frame]
        self._all_frames = frames_ds + additional_frames

        self._update_options()

        return p

    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    def _emalign(self):
        query_map = self._query_map()
        ref_map = self._ref_map()

        if query_map is None:
            self.status('Choose map to align to the reference map.')
            return
        if ref_map is None:
            self.status('Choose map to align the query map to.')
            return
        if query_map == ref_map:
            self.status('The reference map must be different from the query map.')
            return

        downsample = self._get_downsample()
        projections = self._get_projections()
        show_log = self._display_log.value
        use_masking = self._masking.value

        self._run_emalign(ref_map, query_map, downsample, projections, show_log, use_masking)

    def _run_emalign(self, ref_map, query_map, downsample, projections, show_log, use_masking):
        emalign(self.session, ref_map, query_map, downsample=downsample, projections=projections, show_log=show_log, refine=self._use_fit_map.value, mask=use_masking)

    @classmethod
    def get_singleton(self, session, create=True):
        return tools.get_singleton(session, EMalignDialog, 'EMalign', create=create)

    def status(self, message, log=False):
        self._status_label.setText(message)
        if log:
            self.session.logger.info(message)

    def _object_chosen(self):
        self._update_options()
        self._status_label.setText(' ')

    def _update_options(self):
        self._r_map = rm = self._ref_map()
        self._q_map = qm = self._query_map()

        self._check_disable_options()
        if rm is None or qm is None:
            return

        self._r_size = rm.data.size[0]
        self._q_size = qm.data.size[0]
        self._gray_out_downsample_options()
        self._enable_other_options()

    def _check_disable_options(self):
        vlist = self.session.models.list(type=Volume)
        if not vlist:
            for frame in self._all_frames:
                frame.setEnabled(False)

    def _gray_out_downsample_options(self):
        min_v_size = min(self._r_size, self._q_size)
        max_v_size = max(self._r_size, self._q_size)

        if min_v_size < 64:
            ds_values = [True, False, False, False]
        elif min_v_size < 128:
            ds_values = [True, True, False, False]
            self._no_downsample.value = False
            self._downsample_64.value = True
        elif min_v_size < 256:
            ds_values = [True, True, True, False]
        elif min_v_size == 256 and max_v_size == 256:
            ds_values = [True, True, True, True]
        else:
            ds_values = [False, True, True, True]
            if self._no_downsample.value:
                self._downsample_64.value = True
                self._no_downsample.value = False

        if self._r_size != self._q_size:
            ds_values[0] = False

        for i in range(len(self._ds_frames)):
            self._ds_frames[i].setEnabled(ds_values[i])

    def _enable_other_options(self):
        options = [self._projections_frame, self._display_log_frame, self._use_fit_map_frame, self._ds_header_frame, self.header_proj_frame, self._masking_frame]
        for option in options:
            option.setEnabled(True)

    def _ref_map(self):
        m = self._ref_map_menu.value
        return m if isinstance(m, Volume) else None

    # The query map chosen to align to the reference map:
    def _query_map(self):
        m = self._query_map_menu.value
        return m if isinstance(m, Volume) else None

    def _get_downsample(self):
        downsample = 0
        if self._no_downsample.value:
            downsample = None
        if self._downsample_64.value:
            downsample = 64
        if self._downsample_128.value:
            downsample = 128
        if self._downsample_256.value:
            downsample = 256

        return downsample

    def _get_projections(self):
        projections = 0
        if self._projections_25.value:
            projections = 25
        if self._projections_50.value:
            projections = 50
        if self._projections_125.value:
            projections = 125

        return projections


# ----------------------------------------------------------------------------------------------------------------------
def emalign_dialog(session, create=False):
    return EMalignDialog.get_singleton(session, create=create)


def show_emalign_dialog(session):
    return emalign_dialog(session, create=True)
