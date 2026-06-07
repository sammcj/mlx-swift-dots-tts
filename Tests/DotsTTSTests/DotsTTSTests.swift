import XCTest
@testable import DotsTTS

final class DotsTTSTests: XCTestCase {
    func testVersionPresent() {
        XCTAssertFalse(DotsTTS.version.isEmpty)
    }
}
