import glyphUrl from "../../assets/llmedi-glyph.png";

// llmedi stethoscope mark — replaces the upstream Handy hand icon. The file and
// component name are kept so upstream imports stay unchanged.
//
// Cropped from the same brand PNG as the wordmark and used as a CSS mask, so it
// takes the surrounding text color and stays visible in both themes.
const HandyHand = ({
  width,
  height,
  className,
}: {
  width?: number | string;
  height?: number | string;
  className?: string;
}) => (
  <span
    role="img"
    aria-label="llmedi"
    className={className}
    style={{
      display: "inline-block",
      width: width ?? 126,
      height: height ?? 126,
      backgroundColor: "currentColor",
      maskImage: `url(${glyphUrl})`,
      WebkitMaskImage: `url(${glyphUrl})`,
      maskSize: "contain",
      WebkitMaskSize: "contain",
      maskRepeat: "no-repeat",
      WebkitMaskRepeat: "no-repeat",
      maskPosition: "center",
      WebkitMaskPosition: "center",
    }}
  />
);

export default HandyHand;
