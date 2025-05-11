from typing import Optional
from browser_use.browser.context import BrowserSession, BrowserContext, BrowserContextConfig
from playwright.async_api import Page, BrowserContext as PlaywrightContext

class ExtendedBrowserSession(BrowserSession):
    """Extended version of BrowserSession that includes current_page"""
    def __init__(
        self,
        context: PlaywrightContext,
        cached_state: Optional[dict] = None,
        current_page: Optional[Page] = None
    ):
        super().__init__(context=context, cached_state=cached_state)
        self.current_page = current_page


class UseBrowserlessContext(BrowserContext):
    async def _initialize_session(self) -> ExtendedBrowserSession:
        """Initialize a browser session using existing Browserless page.

        Returns:
            ExtendedBrowserSession: The initialized browser session with current page.
        """
        playwright_browser = await self.browser.get_playwright_browser()
        context = await self._create_context(playwright_browser)
        self._add_new_page_listener(context)

        self.session = ExtendedBrowserSession(
            context=context,
            cached_state=None,
        )

        # Get existing page or create new one
        self.session.current_page = context.pages[0] if context.pages else await context.new_page()

        # Initialize session state
        self.session.cached_state = await self._update_state()

        return self.session