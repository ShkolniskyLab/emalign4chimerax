# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and override the method for registering commands,
# inheriting all other methods from the base class.
class _EMalignBundle(BundleAPI):
    # Override methods
    @staticmethod
    def start_tool(session, tool_name):
        from .emalign_gui import show_emalign_dialog
        d = show_emalign_dialog(session)
        return d

    @staticmethod
    def register_command(command_name, logger):
        from . import emalign_cmd
        emalign_cmd.register_emalign_command(logger)


# Create the ``bundle_api`` object that ChimeraX expects:
bundle_api = _EMalignBundle()
