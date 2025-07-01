from django.apps import AppConfig

class CoreConfig(AppConfig):
    name = 'core'  # Nom de la teva aplicació
    verbose_name = 'Gestió d\'Usuaris i Funcions Internes'

    def ready(self):
        # Registra els senyals quan l'aplicació s'ha carregat
        import core.signals
